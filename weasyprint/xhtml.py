"""Semantic fixed-layout XHTML export.

This module deliberately exports a small, versioned interchange format rather
than a complete annual report.  A WeasyPrint :class:`~weasyprint.Document`
usually represents one report subsection; callers can concatenate the page
sections and styles from multiple manifests after their final pagination is
known.

Visible content is taken from the final page box tree.  Text therefore keeps
the line breaking and coordinates used by the PDF, while source elements are
used only to recover logical reading order and semantics.  No source CSS class,
HTML identifier, ``data-*`` attribute or executable content is copied.

Links to anchors rendered by another fragment use opaque tokens in page XHTML.
Their source destinations live only in the intermediate JSON manifest, letting
an assembler reconnect them after globally rebasing ids without exposing the
source names in a public filing.
"""

import base64
import copy
import json
import math
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from tinycss2.color4 import parse_color

from .formatting_structure import boxes
from .images import LinearGradient, RadialGradient, RasterImage, SVGImage
from .matrix import Matrix
from .svg.utils import size as svg_size
from .svg.utils import transform as svg_transform
from .urls import URLFetchingError, fetch

XHTML_NAMESPACE = 'http://www.w3.org/1999/xhtml'
SVG_NAMESPACE = 'http://www.w3.org/2000/svg'
XLINK_NAMESPACE = 'http://www.w3.org/1999/xlink'
FRAGMENT_FORMAT = 'weasyprint.xhtml-fragment'
FRAGMENT_VERSION = 1

_XHTML = f'{{{XHTML_NAMESPACE}}}'
_SEMANTIC_ROOTS = frozenset((
    'address', 'article', 'aside', 'blockquote', 'caption', 'dd', 'details',
    'div', 'dt', 'figcaption', 'figure', 'footer', 'header', 'h1', 'h2',
    'h3', 'h4', 'h5', 'h6', 'li', 'main', 'nav', 'p', 'pre', 'section',
    'summary'))
_PHRASING_TAGS = frozenset((
    'a', 'abbr', 'b', 'cite', 'code', 'del', 'em', 'i', 'ins', 'kbd', 'mark',
    'q', 's', 'samp', 'small', 'strong', 'sub', 'sup', 'time', 'u', 'var'))
_FORBIDDEN_ELEMENTS = frozenset((
    'animate', 'animatecolor', 'animatemotion', 'animatetransform', 'applet',
    'audio', 'canvas', 'discard', 'embed', 'foreignobject', 'frame', 'frameset',
    'iframe', 'object', 'script', 'set', 'source', 'track', 'video'))
_ALLOWED_LINK_SCHEMES = frozenset(('http', 'https', 'mailto', 'tel'))
_ALLOWED_IMAGE_MIMES = frozenset((
    'image/gif', 'image/jpeg', 'image/png', 'image/svg+xml'))
_URL_PATTERN = re.compile(r'url\(\s*(["\']?)(.*?)\1\s*\)', re.I | re.S)
_UNSAFE_CSS_PATTERN = re.compile(
    r'@import|expression\s*\(|javascript\s*:|vbscript\s*:|behavior\s*:',
    re.I)
ElementTree.register_namespace('', XHTML_NAMESPACE)


def _local_name(name):
    """Return a lower-case local name for an XML name."""
    if not isinstance(name, str):
        return ''
    return name.rsplit('}', 1)[-1].lower()


def _number(value):
    """Serialize a finite CSS-pixel value without floating point noise."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '0'
    if not math.isfinite(value):
        return '0'
    value = round(value, 6)
    if value == 0:
        return '0'
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def _px(value):
    return f'{_number(value)}px'


def _css_string(value):
    """Quote an arbitrary value as a CSS string."""
    value = str(value).replace('\\', '\\\\').replace('"', '\\"')
    value = value.replace('\n', '\\a ').replace('\r', '')
    return f'"{value}"'


def _color(value):
    """Serialize a tinycss color to browser CSS."""
    try:
        srgb = value.to('srgb')
        red, green, blue = srgb.coordinates
        alpha = value.alpha
    except (AttributeError, TypeError, ValueError):
        return 'transparent'
    channels = [
        max(0, min(255, round(channel * 255)))
        for channel in (red, green, blue)]
    alpha = max(0, min(1, float(alpha)))
    if alpha == 1:
        return '#{:02x}{:02x}{:02x}'.format(*channels)
    return f'rgba({channels[0]},{channels[1]},{channels[2]},{_number(alpha)})'


def _style_declarations(declarations):
    return ';'.join(
        f'{name}:{value}' for name, value in declarations if value is not None)


def _line_height(style, fallback):
    """Return the used CSS line height in pixels."""
    value = style['line_height']
    if isinstance(value, tuple) and len(value) == 2:
        unit, number = value
        if unit == 'PIXELS':
            return max(0, number)
        if unit == 'NUMBER':
            return max(0, number * style['font_size'])
    return max(0, fallback)


def _is_transparent(value):
    try:
        return value.alpha == 0
    except AttributeError:
        return True


def _safe_external_link(value, base_url=None):
    """Return a safe absolute/user hyperlink, or ``None``."""
    value = (value or '').strip()
    if not value:
        return None
    if value.startswith('#'):
        return value
    absolute = urljoin(base_url or '', value)
    if urlparse(absolute).scheme.lower() in _ALLOWED_LINK_SCHEMES:
        return absolute
    return None


def _accessible_label(value):
    """Normalize accessible source text without applying template policy."""
    return ' '.join((value or '').split())


def _safe_svg_style(value, id_map):
    """Keep SVG CSS only when it is local and non-executable."""
    if not value or _UNSAFE_CSS_PATTERN.search(value):
        return ''

    def replace_url(match):
        url = match.group(2).strip()
        if not url.startswith('#'):
            return 'none'
        mapped = id_map.get(url[1:])
        return f'url(#{mapped})' if mapped else 'none'

    value = _URL_PATTERN.sub(replace_url, value)
    for source, target in id_map.items():
        value = value.replace(f'#{source}', f'#{target}')
    return value


class _Exporter:
    """Export one rendered document to a fragment manifest."""

    def __init__(self, document, fragment_id):
        self.document = document
        self.html = getattr(document, '_html', None)
        self.base_url = getattr(self.html, 'base_url', None)
        self.root = getattr(self.html, 'etree_element', None)
        self.parent = {}
        self.source_order = {}
        if self.root is not None:
            for order, element in enumerate(self.root.iter()):
                self.source_order[id(element)] = order
                for child in element:
                    self.parent[id(child)] = element

        seed = str(fragment_id or self._content_seed()).encode()
        self.prefix = f'xf-{sha256(seed).hexdigest()[:12]}'
        self.anchor_ids = {}
        self.anchor_pages = {}
        self.emitted_anchor_ids = set()
        self.cross_fragment_links = {}
        self.generated_id = 0
        self.stats = defaultdict(int)
        self.warnings = []
        self._prepare_anchors()

    def _content_seed(self):
        title = self.document.metadata.title or ''
        lang = self.document.metadata.lang or ''
        dimensions = '|'.join(
            f'{_number(page.width)}x{_number(page.height)}'
            for page in self.document.pages)
        sample = []
        sample_length = 0
        for page in self.document.pages:
            for box in page._page_box.descendants(placeholders=True):
                if isinstance(box, boxes.TextBox):
                    sample.append(box.text)
                    sample_length += len(box.text)
                    if sample_length >= 65536:
                        break
            if sample_length >= 65536:
                break
        return f'{title}|{lang}|{dimensions}|{"".join(sample)}'

    def _prepare_anchors(self):
        names = []
        seen = set()
        if self.root is not None:
            for element in self.root.iter():
                name = element.get('id')
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
        for page in self.document.pages:
            for name in page.anchors:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        for index, name in enumerate(names, 1):
            self.anchor_ids[name] = f'{self.prefix}-a{index}'

        for page_number, page in enumerate(self.document.pages, 1):
            for name in page.anchors:
                self.anchor_pages.setdefault(name, page_number)

    def export(self):
        pages = []
        for number, page in enumerate(self.document.pages, 1):
            page_id = f'{self.prefix}-p{number}'
            element = self._page_element(page, page_id, number)
            pages.append({
                'id': page_id,
                'number': number,
                'width': round(float(page.width), 6),
                'height': round(float(page.height), 6),
                'xhtml': ElementTree.tostring(
                    element, encoding='unicode', short_empty_elements=True),
            })

        anchors = [
            {
                'source': source,
                'target': target,
                'page': self.anchor_pages.get(source),
            }
            for source, target in self.anchor_ids.items()
            if target in self.emitted_anchor_ids
        ]
        links = [
            {'token': token, 'source': source}
            for source, token in self.cross_fragment_links.items()
        ]
        manifest = {
            'format': FRAGMENT_FORMAT,
            'version': FRAGMENT_VERSION,
            'css_unit': 'px',
            'fragment_id': self.prefix,
            'title': self.document.metadata.title,
            'lang': self.document.metadata.lang,
            'stylesheet': self._stylesheet(),
            'pages': pages,
            'anchors': anchors,
            'links': links,
            'statistics': dict(sorted(self.stats.items())),
            'warnings': self.warnings,
        }
        return manifest

    def _stylesheet(self):
        rules = [
            '.wp-page{position:relative;display:block;overflow:hidden;'
            'margin:0;box-sizing:border-box;isolation:isolate}',
            '.wp-decoration{position:absolute;display:block;box-sizing:border-box;'
            'pointer-events:none}',
            '.wp-text{position:absolute;display:block;box-sizing:border-box;'
            'margin:0;padding:0;border:0;white-space:pre;transform-origin:0 0}',
            '.wp-image{position:absolute;display:block;margin:0;padding:0;'
            'border:0;max-width:none;max-height:none;object-fit:fill}',
            '.wp-anchor{position:absolute;display:block;width:0;height:0;'
            'overflow:hidden}',
            '.wp-semantic{position:static;margin:0;padding:0;border:0;'
            'font:inherit;color:inherit}',
            '.wp-table{position:static;border-collapse:collapse;border:0;'
            'margin:0;padding:0;width:0;height:0;table-layout:fixed}',
            '.wp-table thead,.wp-table tbody,.wp-table tfoot,.wp-table tr,'
            '.wp-table th,.wp-table td{position:static;margin:0;padding:0;'
            'border:0;width:0;height:0;min-width:0;min-height:0;'
            'font:inherit;color:inherit}',
        ]
        rules.extend(self._font_rules())
        return ''.join(rules)

    def _font_rules(self):
        rules = []
        seen = set()
        for font in self.document.fonts.values():
            content = getattr(font, 'file_content', None)
            if not content:
                continue
            digest = sha256(content).hexdigest()
            key = (
                digest, getattr(font, 'family', ''),
                getattr(font, 'weight', 400), getattr(font, 'style', 0))
            if key in seen:
                continue
            seen.add(key)
            if content[:4] == b'OTTO':
                mime, font_format = 'font/otf', 'opentype'
            elif content[:4] == b'wOFF':
                mime, font_format = 'font/woff', 'woff'
            elif content[:4] == b'wOF2':
                mime, font_format = 'font/woff2', 'woff2'
            else:
                mime, font_format = 'font/ttf', 'truetype'
            encoded = base64.b64encode(content).decode('ascii')
            style = {0: 'normal', 1: 'oblique', 2: 'italic'}.get(
                getattr(font, 'style', 0), 'normal')
            rules.append(
                '@font-face{font-family:' + _css_string(font.family) +
                f';font-style:{style};font-weight:{font.weight};font-display:block;'
                f'src:url("data:{mime};base64,{encoded}") format("{font_format}")' +
                '}')
            self.stats['embedded_fonts'] += 1
        if not rules:
            self.warnings.append(
                'No fonts were embedded. Call write_pdf before XHTML export '
                'to populate the rendered document font cache.')
        return rules

    def _page_element(self, page, page_id, number):
        self.current_page = number
        section = ElementTree.Element(f'{_XHTML}section', {
            'class': 'wp-page',
            'id': page_id,
            'aria-label': f'Page {number}',
            'style': _style_declarations((
                ('width', _px(page.width)), ('height', _px(page.height)))),
        })
        decoration_layer = ElementTree.SubElement(
            section, f'{_XHTML}div', {
                'class': 'wp-semantic', 'aria-hidden': 'true'})
        content_layer = ElementTree.SubElement(
            section, f'{_XHTML}div', {'class': 'wp-semantic'})

        self._add_decorations(page._page_box, decoration_layer)
        table_boxes, primitive_records = self._collect_content(page._page_box)
        table_elements = {
            id(table): self._table_element(table, hidden)
            for table, hidden in table_boxes}

        groups = defaultdict(list)
        group_elements = {}
        group_hidden = {}
        for record in primitive_records:
            box, hidden, opacity = record
            root = self._semantic_root(box.element)
            key = (id(root), hidden) if root is not None else (id(box), hidden)
            groups[key].append((box, opacity))
            group_elements[key] = root
            group_hidden[key] = hidden

        entries = []
        for table, hidden in table_boxes:
            source_order = self.source_order.get(id(table.element))
            if source_order is None:
                order = (1, 0, table.position_y, table.position_x)
            else:
                order = (0, source_order, 0, 0)
            entries.append((order, 0, ('table', id(table))))
        for key, records in groups.items():
            element = group_elements[key]
            source_order = self.source_order.get(id(element))
            if source_order is None:
                first_box = min(
                    (box for box, _opacity in records),
                    key=lambda box: (box.position_y, box.position_x))
                order = (1, 0, first_box.position_y, first_box.position_x)
            else:
                order = (0, source_order, 0, 0)
            entries.append((order, 1, ('group', key)))
        entries.sort(key=lambda item: (item[0], item[1]))

        emitted_lists = set()
        for _order, _kind_order, (kind, key) in entries:
            if kind == 'table':
                content_layer.append(table_elements[key])
                continue
            root = group_elements[key]
            if root is not None and _local_name(root.tag) == 'li':
                list_element = self.parent.get(id(root))
                if _local_name(getattr(list_element, 'tag', '')) in ('ol', 'ul'):
                    list_key = (id(list_element), group_hidden[key])
                    if list_key in emitted_lists:
                        continue
                    emitted_lists.add(list_key)
                    tag = _local_name(list_element.tag)
                    wrapper = ElementTree.Element(
                        f'{_XHTML}{tag}', {'class': 'wp-semantic'})
                    if group_hidden[key]:
                        wrapper.set('aria-hidden', 'true')
                    child_keys = [
                        candidate for candidate in groups
                        if group_elements[candidate] is not None and
                        self.parent.get(id(group_elements[candidate])) is
                        list_element and
                        group_hidden[candidate] == group_hidden[key]]
                    child_keys.sort(key=lambda candidate: self.source_order.get(
                        id(group_elements[candidate]), 10 ** 12))
                    for child_key in child_keys:
                        wrapper.append(self._semantic_group(
                            group_elements[child_key], groups[child_key],
                            group_hidden[child_key]))
                    content_layer.append(wrapper)
                    continue
            content_layer.append(self._semantic_group(
                root, groups[key], group_hidden[key]))

        # Some anchors belong to layout-only containers that are not useful
        # semantic wrappers.  Emit zero-size targets at WeasyPrint's resolved
        # coordinates so that every advertised target exists exactly once.
        for source, target_rectangle in page.anchors.items():
            x, y = target_rectangle[:2]
            target = self.anchor_ids.get(source)
            if not target or target in self.emitted_anchor_ids:
                continue
            ElementTree.SubElement(content_layer, f'{_XHTML}span', {
                'class': 'wp-anchor',
                'id': target,
                'aria-hidden': 'true',
                'style': _style_declarations((
                    ('left', _px(x)), ('top', _px(y)))),
            })
            self.emitted_anchor_ids.add(target)
            self.anchor_pages[source] = number

        self.stats['pages'] += 1
        return section

    def _collect_content(self, root_box):
        table_boxes = []
        primitives = []

        def walk(box, hidden=False, opacity=1, inside_table=False):
            hidden = hidden or getattr(box, 'aria_hidden', False)
            hidden = hidden or isinstance(box, boxes.MarginBox)
            try:
                opacity *= box.style['opacity']
            except (KeyError, TypeError):
                pass
            if isinstance(box, boxes.TableBox) and not inside_table:
                table_boxes.append((box, hidden))
                return
            if isinstance(box, (boxes.TextBox, boxes.ReplacedBox)):
                primitives.append((box, hidden, opacity))
                return
            if isinstance(box, boxes.ParentBox):
                for child in box.children:
                    if not isinstance(child, boxes.Box):
                        child = getattr(child, '_box', None)
                    if isinstance(child, boxes.Box):
                        walk(child, hidden, opacity, inside_table)

        walk(root_box)
        return table_boxes, primitives

    def _semantic_root(self, element):
        candidate = None
        while element is not None:
            tag = _local_name(element.tag)
            if tag in _SEMANTIC_ROOTS:
                if tag not in ('div', 'section'):
                    return element
                if candidate is None:
                    candidate = element
            element = self.parent.get(id(element))
        return candidate

    def _semantic_group(self, source, records, hidden):
        tag = _local_name(source.tag) if source is not None else 'div'
        if tag not in _SEMANTIC_ROOTS:
            tag = 'div'
        element = ElementTree.Element(
            f'{_XHTML}{tag}', {'class': 'wp-semantic'})
        if hidden:
            element.set('aria-hidden', 'true')
        self._copy_semantic_attributes(source, element)
        for box, opacity in records:
            primitive = self._primitive_element(box, opacity)
            if primitive is not None:
                element.append(self._wrap_phrasing(box.element, source, primitive))
        self.stats[f'{tag}_elements'] += 1
        return element

    def _wrap_phrasing(self, source, root, primitive):
        wrappers = []
        element = source
        while element is not None and element is not root:
            tag = _local_name(element.tag)
            if tag in _PHRASING_TAGS:
                wrappers.append(element)
            element = self.parent.get(id(element))
        for source_wrapper in wrappers:
            tag = _local_name(source_wrapper.tag)
            wrapper = ElementTree.Element(
                f'{_XHTML}{tag}', {'class': 'wp-semantic'})
            self._copy_semantic_attributes(source_wrapper, wrapper)
            wrapper.append(primitive)
            primitive = wrapper
        return primitive

    def _copy_semantic_attributes(self, source, target):
        if source is None:
            return
        source_id = self._nearest_anchor(source)
        if source_id in self.anchor_ids:
            identifier = self.anchor_ids[source_id]
            if identifier not in self.emitted_anchor_ids:
                target.set('id', identifier)
                self.emitted_anchor_ids.add(identifier)
                self.anchor_pages.setdefault(source_id, self.current_page)
        tag = _local_name(source.tag)
        if tag != 'img':
            label = _accessible_label(source.get('aria-label'))
            if label:
                target.set('aria-label', label)
        if tag == 'a':
            href = _safe_external_link(source.get('href'), self.base_url)
            if href and href.startswith('#'):
                source_target = href[1:]
                if not source_target:
                    href = None
                elif source_target in self.anchor_pages:
                    href = f'#{self.anchor_ids[source_target]}'
                else:
                    token = self.cross_fragment_links.get(source_target)
                    if token is None:
                        token = (
                            f'{self.prefix}-l{len(self.cross_fragment_links) + 1}')
                        self.cross_fragment_links[source_target] = token
                    href = f'#{token}'
            if href:
                target.set('href', href)
            if source.get('title'):
                target.set('title', source.get('title'))
        if tag == 'li':
            current = source.get('aria-current', '').lower()
            if current in ('page', 'step', 'location', 'date', 'time', 'true'):
                target.set('aria-current', current)
        if tag in ('th', 'td'):
            scope = source.get('scope', '').lower()
            if scope in ('col', 'colgroup', 'row', 'rowgroup'):
                target.set('scope', scope)

    def _table_element(self, table, hidden):
        attributes = {'class': 'wp-table'}
        if hidden:
            attributes['aria-hidden'] = 'true'
        element = ElementTree.Element(f'{_XHTML}table', attributes)
        self._copy_semantic_attributes(table.element, element)
        label = None
        if table.element is not None:
            label = table.element.get('aria-label') or table.element.get('title')
        if label:
            element.set('aria-label', label)

        cell_ids = {}
        source_to_local = {}
        for descendant in table.descendants(placeholders=True):
            if not isinstance(descendant, boxes.TableCellBox):
                continue
            source = descendant.element
            source_id = source.get('id') if source is not None else None
            if not source_id:
                continue
            anchor_id = self.anchor_ids.get(source_id)
            if anchor_id and anchor_id not in self.emitted_anchor_ids:
                identifier = anchor_id
                self.emitted_anchor_ids.add(identifier)
                self.anchor_pages.setdefault(source_id, self.current_page)
            else:
                self.generated_id += 1
                identifier = f'{self.prefix}-c{self.generated_id}'
            cell_ids[id(descendant)] = identifier
            source_to_local.setdefault(source_id, identifier)

        emitted_cells = []

        def add_rows(parent, children):
            for child in children:
                if isinstance(child, boxes.TableRowGroupBox):
                    tag = _local_name(getattr(child.element, 'tag', ''))
                    if tag not in ('thead', 'tbody', 'tfoot'):
                        tag = 'thead' if child.is_header else (
                            'tfoot' if child.is_footer else 'tbody')
                    group = ElementTree.SubElement(
                        parent, f'{_XHTML}{tag}', {'class': 'wp-semantic'})
                    add_rows(group, child.children)
                elif isinstance(child, boxes.TableRowBox):
                    row = ElementTree.SubElement(
                        parent, f'{_XHTML}tr', {'class': 'wp-semantic'})
                    for cell_box in child.children:
                        if not isinstance(cell_box, boxes.TableCellBox):
                            continue
                        cell_tag = _local_name(getattr(cell_box.element, 'tag', ''))
                        if cell_tag not in ('td', 'th'):
                            cell_tag = 'td'
                        cell = ElementTree.SubElement(
                            row, f'{_XHTML}{cell_tag}', {'class': 'wp-semantic'})
                        self._table_cell_attributes(
                            cell_box, cell, cell_ids, source_to_local)
                        emitted_cells.append((
                            cell,
                            cell_box.grid_x,
                            cell_box.colspan,
                        ))
                        for descendant in cell_box.descendants(placeholders=True):
                            if descendant is cell_box:
                                continue
                            if isinstance(descendant, boxes.TableBox):
                                continue
                            if isinstance(
                                    descendant,
                                    (boxes.TextBox, boxes.ReplacedBox)):
                                primitive = self._primitive_element(descendant, 1)
                                if primitive is not None:
                                    cell.append(primitive)
                elif isinstance(child, boxes.TableBox):
                    add_rows(parent, child.children)

        add_rows(element, table.children)
        # Paginated table fragments can retain source colspans for columns
        # that have no cell beginning in them on this page.  These empty
        # layout-only columns make the fragment's HTML table model invalid and
        # carry no data.  Compress them while preserving every actual cell and
        # the spans across columns that do contain cells.
        starts = {grid_x for _cell, grid_x, _colspan in emitted_cells}
        for cell, grid_x, colspan in emitted_cells:
            semantic_span = sum(
                grid_x <= start < grid_x + colspan for start in starts)
            if semantic_span > 1:
                cell.set('colspan', str(semantic_span))
            else:
                cell.attrib.pop('colspan', None)
        self.stats['tables'] += 1
        return element

    def _table_cell_attributes(self, box, target, cell_ids, source_to_local):
        source = box.element
        if box.colspan > 1:
            target.set('colspan', str(box.colspan))
        if box.rowspan != 1:
            target.set('rowspan', str(box.rowspan))
        if source is None:
            return
        if identifier := cell_ids.get(id(box)):
            target.set('id', identifier)
        headers = []
        for header in source.get('headers', '').split():
            if header in source_to_local:
                headers.append(source_to_local[header])
        if headers:
            target.set('headers', ' '.join(headers))
        scope = source.get('scope', '').lower()
        if scope in ('col', 'colgroup', 'row', 'rowgroup'):
            target.set('scope', scope)

    def _primitive_element(self, box, opacity):
        if isinstance(box, boxes.TextBox):
            return self._text_element(box, opacity)
        if isinstance(box, boxes.ReplacedBox):
            return self._image_element(box, opacity)

    def _text_element(self, box, opacity):
        style = box.style
        family = ','.join(_css_string(name) for name in style['font_family'])
        letter_spacing = style['letter_spacing']
        if letter_spacing == 'normal':
            letter_spacing = 'normal'
        else:
            letter_spacing = _px(letter_spacing)
        decoration = style['text_decoration_line']
        if isinstance(decoration, (tuple, list, set)):
            decoration = ' '.join(decoration) or 'none'
        declarations = [
            ('left', _px(box.position_x)),
            ('top', _px(box.position_y)),
            ('width', _px(max(0, box.width))),
            ('height', _px(max(0, box.height))),
            ('font-family', family),
            ('font-size', _px(style['font_size'])),
            ('font-style', style['font_style']),
            ('font-weight', str(style['font_weight'])),
            ('font-stretch', style['font_stretch']),
            # TextBox.height is the font-height, not the line-height used to
            # paint its baseline.  Keeping the computed line height preserves
            # vertical alignment in tall lines without template-specific
            # positioning fixes.
            ('line-height', _px(_line_height(style, box.height))),
            ('letter-spacing', letter_spacing),
            ('color', _color(style['color'])),
            ('text-decoration-line', decoration),
            ('text-decoration-style', style['text_decoration_style']),
            ('text-decoration-color', _color(style['text_decoration_color'])),
            ('direction', style['direction']),
            ('unicode-bidi', style['unicode_bidi']),
            ('opacity', _number(opacity) if opacity < 1 else None),
        ]
        if box.transformation_matrix:
            values = ','.join(
                _number(value) for value in box.transformation_matrix.values)
            declarations.append(('transform', f'matrix({values})'))
        element = ElementTree.Element(f'{_XHTML}span', {
            'class': 'wp-text', 'style': _style_declarations(declarations)})
        element.text = box.text
        self.stats['text_boxes'] += 1
        self.stats['text_characters'] += len(box.text)
        return element

    def _image_element(self, box, opacity):
        svg_text = []
        if isinstance(box.replacement, SVGImage):
            svg_text = self._svg_text(box.replacement, box)
            uri = self._svg_data_uri(box.replacement, remove_text=bool(svg_text))
        else:
            uri = self._image_data_uri(box.replacement)
        if not uri:
            self.warnings.append(
                f'Unsupported image on page at ({_number(box.position_x)}, '
                f'{_number(box.position_y)})')
            return None
        declarations = [
            ('left', _px(box.content_box_x())),
            ('top', _px(box.content_box_y())),
            ('width', _px(max(0, box.width))),
            ('height', _px(max(0, box.height))),
            ('opacity', _number(opacity) if opacity < 1 else None),
        ]
        if box.transformation_matrix:
            values = ','.join(
                _number(value) for value in box.transformation_matrix.values)
            declarations.append(('transform', f'matrix({values})'))
            declarations.append(('transform-origin', '0 0'))
        alt = self._image_alt(box.element, svg_text)
        element = ElementTree.Element(f'{_XHTML}img', {
            'class': 'wp-image',
            'style': _style_declarations(declarations),
            'src': uri,
            'alt': '' if svg_text else alt,
        })
        if svg_text:
            element.set('aria-hidden', 'true')
        self.stats['images'] += 1
        if not svg_text:
            return element
        wrapper = ElementTree.Element(
            f'{_XHTML}span', {'class': 'wp-semantic'})
        wrapper.append(element)
        for text, declarations in svg_text:
            overlay = ElementTree.SubElement(wrapper, f'{_XHTML}span', {
                'class': 'wp-text',
                'style': _style_declarations(declarations),
            })
            overlay.text = text
            self.stats['svg_text_overlays'] += 1
            self.stats['text_characters'] += len(text)
        return wrapper

    def _image_alt(self, element, svg_text=()):
        source_alt = ''
        current = element
        while current is not None:
            for attribute in ('aria-label', 'alt'):
                source_alt = _accessible_label(current.get(attribute))
                if source_alt:
                    break
            if source_alt:
                break
            current = self.parent.get(id(current))
        labels = [text for text, _declarations in svg_text]
        if labels:
            chart_text = ', '.join(labels)
            return f'{source_alt}: {chart_text}' if source_alt else chart_text
        return source_alt

    def _image_data_uri(self, image):
        if isinstance(image, RasterImage):
            mime = 'image/jpeg' if image.format == 'JPEG' else 'image/png'
            return self._data_uri(mime, image.image_data.data)
        if isinstance(image, SVGImage):
            return self._svg_data_uri(image)
        if isinstance(image, (LinearGradient, RadialGradient)):
            return None
        return None

    def _svg_data_uri(self, image, remove_text=False):
        root = getattr(image, '_xhtml_source', image._svg.tree._etree_node)
        content = self._sanitize_svg(root, remove_text=remove_text)
        return self._data_uri('image/svg+xml', content)

    def _svg_text(self, image, box):
        root = getattr(image, '_xhtml_source', image._svg.tree._etree_node)
        view_box = root.get('viewBox', '').replace(',', ' ').split()
        try:
            min_x, min_y, view_width, view_height = map(float, view_box)
        except (TypeError, ValueError):
            return []
        if view_width <= 0 or view_height <= 0:
            return []

        scale_x = box.width / view_width
        scale_y = box.height / view_height
        aspect_ratio = root.get('preserveAspectRatio', 'xMidYMid').split()
        if aspect_ratio and aspect_ratio[0].lower() == 'defer':
            aspect_ratio.pop(0)
        align = aspect_ratio[0] if aspect_ratio else 'xMidYMid'
        if align.lower() != 'none':
            meet_or_slice = aspect_ratio[1].lower() if len(
                aspect_ratio) > 1 else 'meet'
            scale = (
                max(scale_x, scale_y) if meet_or_slice == 'slice' else
                min(scale_x, scale_y))
            scale_x = scale_y = scale
            remaining_x = box.width - view_width * scale
            remaining_y = box.height - view_height * scale
            lower_align = align.lower()
            align_x = 0.5 if 'xmid' in lower_align else (
                1 if 'xmax' in lower_align else 0)
            align_y = 0.5 if 'ymid' in lower_align else (
                1 if 'ymax' in lower_align else 0)
        else:
            remaining_x = remaining_y = 0
            align_x = align_y = 0
        viewport = Matrix(
            a=scale_x, d=scale_y,
            e=(box.content_box_x() + remaining_x * align_x -
               min_x * scale_x),
            f=(box.content_box_y() + remaining_y * align_y -
               min_y * scale_y))
        normalized_diagonal = math.hypot(
            view_width, view_height) / math.sqrt(2)
        records = []

        def walk(element, parent_matrix, inherited, parent_font_size):
            direct = {}
            for declaration in element.get('style', '').split(';'):
                if ':' in declaration:
                    name, value = declaration.split(':', 1)
                    direct[name.strip().lower()] = value.strip()
            for name in (
                    'display', 'fill', 'font-family', 'font-size',
                    'font-style', 'font-weight', 'opacity', 'text-anchor',
                    'visibility'):
                if element.get(name) is not None:
                    direct.setdefault(name, element.get(name))
            properties = inherited.copy()
            properties.update(direct)

            font_size_value = direct.get('font-size')
            if font_size_value is None:
                font_size = parent_font_size
            else:
                try:
                    font_size = svg_size(
                        font_size_value, parent_font_size, parent_font_size)
                except (AssertionError, TypeError, ValueError):
                    font_size = parent_font_size
                if not math.isfinite(font_size) or font_size <= 0:
                    font_size = parent_font_size

            transform_string = direct.get(
                'transform', element.get('transform', ''))
            transform_origin = direct.get(
                'transform-origin', element.get('transform-origin', '0 0'))
            try:
                local_matrix = svg_transform(
                    transform_string, transform_origin, font_size,
                    normalized_diagonal)
            except (AssertionError, IndexError, TypeError, ValueError):
                local_matrix = Matrix()
            matrix = local_matrix @ parent_matrix

            if _local_name(element.tag) == 'text':
                record = self._svg_text_record(
                    element, properties, font_size, matrix @ viewport,
                    view_width, view_height)
                if record is not None:
                    records.append(record)

            for child in element:
                walk(child, matrix, properties, font_size)

        walk(root, Matrix(), {}, 12)
        return records

    def _svg_text_record(self, text_element, properties, font_size, matrix,
                         view_width, view_height):
        text = ''.join(text_element.itertext()).strip()
        if not text:
            return None

        def coordinate(name, reference):
            value = text_element.get(name, '0')
            try:
                return svg_size(value, font_size, reference)
            except (AssertionError, TypeError, ValueError):
                return 0

        x = coordinate('x', view_width) + coordinate('dx', view_width)
        y = coordinate('y', view_height) + coordinate('dy', view_height)
        baseline_x, baseline_y = matrix.transform_point(x, y)
        a, b, c, d, _e, _f = matrix.values
        if not all(math.isfinite(value) for value in (
                baseline_x, baseline_y, a, b, c, d)):
            return None

        family = properties.get('font-family', 'sans-serif')
        family = family.strip().strip('"\'') or 'sans-serif'
        anchor = properties.get('text-anchor', 'start').lower()
        fill = properties.get('fill', '#000')
        parsed_fill = parse_color(fill)
        fill = _color(parsed_fill) if parsed_fill is not None else '#000000'

        transform = []
        axis_aligned = (
            abs(b) <= 1e-9 and abs(c) <= 1e-9 and a > 0 and d > 0)
        if axis_aligned:
            scaled_size = font_size * d
            left = baseline_x
            top = baseline_y - scaled_size
            output_font_size = scaled_size
            horizontal_ratio = a / d
            if not math.isclose(horizontal_ratio, 1, abs_tol=1e-9):
                transform.append(f'scaleX({_number(horizontal_ratio)})')
        else:
            left, top = matrix.transform_point(x, y - font_size)
            output_font_size = font_size
            transform.append(
                'matrix(' + ','.join(_number(value) for value in (
                    a, b, c, d, 0, 0)) + ')')

        # SVG text anchoring happens in the text's local coordinate system,
        # before its transform.  Keeping the percentage translation last in
        # the CSS transform list preserves that order for scaled/rotated text.
        if anchor == 'middle':
            transform.append('translateX(-50%)')
        elif anchor == 'end':
            transform.append('translateX(-100%)')

        font_style = properties.get('font-style', 'normal')
        if font_style not in ('normal', 'italic', 'oblique'):
            font_style = 'normal'
        font_weight = properties.get('font-weight', '400')
        if re.fullmatch(r'(?:normal|bold|[1-9]00)', font_weight) is None:
            font_weight = '400'
        opacity = properties.get('opacity', '1')
        try:
            opacity = float(opacity)
        except ValueError:
            opacity = 1
        declarations = [
            ('left', _px(left)),
            ('top', _px(top)),
            ('width', 'max-content'),
            ('height', _px(output_font_size * 1.2)),
            ('font-family', _css_string(family)),
            ('font-size', _px(output_font_size)),
            ('font-style', font_style),
            ('font-weight', font_weight),
            ('line-height', _px(output_font_size * 1.2)),
            ('color', fill),
            ('opacity', _number(opacity) if opacity < 1 else None),
            ('transform', ' '.join(transform) if transform else None),
            ('transform-origin', '0 0' if transform else None),
        ]
        return text, declarations

    def _data_uri(self, mime, content):
        if mime not in _ALLOWED_IMAGE_MIMES or not content:
            return None
        return f'data:{mime};base64,{base64.b64encode(content).decode("ascii")}'

    def _sanitize_svg(self, source_root, remove_text=False):
        # tinyhtml5 can retain namespace declaration attributes in a form that
        # ElementTree refuses to parse back from its own serialization.  A
        # deep copy preserves the already parsed tree and avoids that lossy
        # round trip.
        root = copy.deepcopy(source_root)
        elements = list(root.iter())
        id_map = {}
        for index, element in enumerate(elements, 1):
            source_id = element.get('id')
            if source_id:
                id_map[source_id] = f'v{index:x}'

        for parent in list(root.iter()):
            for child in list(parent):
                child_name = _local_name(child.tag)
                if child_name in _FORBIDDEN_ELEMENTS or (
                        remove_text and child_name == 'text'):
                    parent.remove(child)
        for element in root.iter():
            source_id = element.get('id')
            for name in list(element.attrib):
                if name.startswith('{http://www.w3.org/2000/xmlns/}'):
                    # tinyhtml5 exposes namespace declarations as ordinary
                    # attributes.  ElementTree writes the required namespace
                    # declarations itself; retaining these creates invalid
                    # reserved-prefix bindings.
                    del element.attrib[name]
                    continue
                local = _local_name(name)
                value = element.attrib[name]
                if local.startswith(('on', 'data-')):
                    del element.attrib[name]
                elif local == 'class':
                    del element.attrib[name]
                elif local == 'id':
                    if source_id in id_map:
                        element.attrib[name] = id_map[source_id]
                    else:
                        del element.attrib[name]
                elif local in ('href', 'src'):
                    if value.startswith('#'):
                        mapped = id_map.get(value[1:])
                        if mapped:
                            element.attrib[name] = f'#{mapped}'
                        else:
                            del element.attrib[name]
                    elif value.startswith('data:'):
                        if not value.lower().startswith(tuple(
                                f'data:{mime}' for mime in _ALLOWED_IMAGE_MIMES)):
                            del element.attrib[name]
                    else:
                        embedded = self._fetch_image_uri(value)
                        if embedded:
                            element.attrib[name] = embedded
                        else:
                            del element.attrib[name]
                elif local == 'style':
                    safe = _safe_svg_style(value, id_map)
                    if safe:
                        element.attrib[name] = safe
                    else:
                        del element.attrib[name]
                elif 'url(' in value.lower():
                    safe = _safe_svg_style(value, id_map)
                    if safe:
                        element.attrib[name] = safe
                    else:
                        del element.attrib[name]
            if _local_name(element.tag) == 'style':
                element.text = _safe_svg_style(element.text or '', id_map)
        return ElementTree.tostring(root, encoding='utf-8', xml_declaration=False)

    def _nearest_anchor(self, source):
        while source is not None:
            source_id = source.get('id')
            if source_id:
                return source_id
            source = self.parent.get(id(source))

    def _fetch_image_uri(self, uri):
        absolute = urljoin(self.base_url or '', uri)
        try:
            with fetch(self.document.url_fetcher, absolute) as result:
                content = result.get('string')
                if content is None:
                    content = result['file_obj'].read()
                mime = result.get('mime_type', '').split(';', 1)[0].lower()
        except (URLFetchingError, KeyError, OSError):
            return None
        if mime == 'image/svg+xml':
            try:
                content = self._sanitize_svg(ElementTree.fromstring(content))
            except ElementTree.ParseError:
                return None
        return self._data_uri(mime, content)

    def _add_decorations(self, root_box, parent):
        for box in root_box.descendants(placeholders=True):
            self._add_box_decoration(box, parent)

    def _add_box_decoration(self, box, parent):
        background = getattr(box, 'background', None)
        widths = [
            getattr(box, f'border_{side}_width', 0)
            for side in ('top', 'right', 'bottom', 'left')]
        has_border = any(width > 0 for width in widths)
        try:
            column_rule_width = box.style['column_rule_width']
            column_rule_style = box.style['column_rule_style']
            has_column_rule = (
                column_rule_width > 0 and
                column_rule_style not in ('none', 'hidden'))
        except (KeyError, TypeError):
            has_column_rule = False
        if background is None and not has_border and not has_column_rule:
            return
        try:
            if isinstance(box, boxes.PageBox):
                x, y = 0, 0
                width, height = box.margin_width(), box.margin_height()
            else:
                x, y = box.border_box_x(), box.border_box_y()
                width, height = box.border_width(), box.border_height()
        except AttributeError:
            return
        declarations = [
            ('left', _px(x)), ('top', _px(y)),
            ('width', _px(max(0, width))), ('height', _px(max(0, height))),
        ]
        if background is not None and not _is_transparent(background.color):
            declarations.append(('background-color', _color(background.color)))
        for side, border_width in zip(('top', 'right', 'bottom', 'left'), widths):
            if border_width <= 0:
                continue
            style = box.style[f'border_{side}_style']
            color = _color(box.style[f'border_{side}_color'])
            declarations.append((
                f'border-{side}', f'{_px(border_width)} {style} {color}'))
        radii = (
            getattr(box, 'border_top_left_radius', (0, 0)),
            getattr(box, 'border_top_right_radius', (0, 0)),
            getattr(box, 'border_bottom_right_radius', (0, 0)),
            getattr(box, 'border_bottom_left_radius', (0, 0)))
        if any(any(radius) for radius in radii):
            horizontal = ' '.join(_px(radius[0]) for radius in radii)
            vertical = ' '.join(_px(radius[1]) for radius in radii)
            declarations.append((
                'border-radius', f'{horizontal} / {vertical}'))
        try:
            opacity = box.style['opacity']
        except (KeyError, TypeError):
            opacity = 1
        if opacity < 1:
            declarations.append(('opacity', _number(opacity)))
        if box.transformation_matrix:
            values = ','.join(
                _number(value) for value in box.transformation_matrix.values)
            declarations.extend((
                ('transform', f'matrix({values})'), ('transform-origin', '0 0')))
        element = ElementTree.SubElement(parent, f'{_XHTML}div', {
            'class': 'wp-decoration',
            'style': _style_declarations(declarations),
        })
        self.stats['decorations'] += 1

        if has_column_rule and isinstance(box, boxes.ParentBox):
            columns = [
                child for child in box.children
                if isinstance(child, boxes.Box)]
            columns.sort(key=lambda child: child.border_box_x())
            for first, second in zip(columns, columns[1:]):
                first_right = first.border_box_x() + first.border_width()
                second_left = second.border_box_x()
                if second_left <= first_right:
                    continue
                center = (first_right + second_left) / 2
                rule_style = _style_declarations((
                    ('left', _px(center - x)),
                    ('top', _px(box.content_box_y() - y)),
                    ('height', _px(box.height)),
                    ('width', '0'),
                    ('border-left',
                     f'{_px(column_rule_width)} {column_rule_style} '
                     f'{_color(box.style["column_rule_color"])}'),
                ))
                ElementTree.SubElement(element, f'{_XHTML}span', {
                    'class': 'wp-decoration', 'style': rule_style})

        if background is None:
            return
        for layer in reversed(background.layers):
            if layer.image is None or not isinstance(layer.size, tuple):
                continue
            uri = self._image_data_uri(layer.image)
            if not uri:
                self.warnings.append('Unsupported CSS background image omitted')
                continue
            paint_x, paint_y, paint_width, paint_height = layer.painting_area
            position_x, position_y, _position_width, _position_height = (
                layer.positioning_area)
            image_x, image_y = layer.position
            image_width, image_height = layer.size
            repeat_x, repeat_y = layer.repeat
            repeat = (
                'no-repeat' if repeat_x == repeat_y == 'no-repeat' else
                'repeat-x' if repeat_x != 'no-repeat' and repeat_y == 'no-repeat' else
                'repeat-y' if repeat_x == 'no-repeat' and repeat_y != 'no-repeat' else
                'repeat')
            layer_style = _style_declarations((
                ('left', _px(paint_x - x)),
                ('top', _px(paint_y - y)),
                ('width', _px(paint_width)),
                ('height', _px(paint_height)),
                ('background-image', f'url("{uri}")'),
                ('background-size', f'{_px(image_width)} {_px(image_height)}'),
                ('background-position',
                 f'{_px(position_x + image_x - paint_x)} '
                 f'{_px(position_y + image_y - paint_y)}'),
                ('background-repeat', repeat),
            ))
            ElementTree.SubElement(element, f'{_XHTML}span', {
                'class': 'wp-decoration', 'style': layer_style})


def export_xhtml_fragment(document, *, fragment_id=None):
    """Return a JSON-serializable XHTML fragment manifest."""
    return _Exporter(document, fragment_id).export()


def write_xhtml_fragment(document, target=None, *, fragment_id=None):
    """Write an XHTML fragment manifest and mirror ``Document.write_pdf``."""
    manifest = export_xhtml_fragment(document, fragment_id=fragment_id)
    output = json.dumps(
        manifest, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    if target is None:
        return output
    if hasattr(target, 'write'):
        target.write(output)
    else:
        Path(target).write_bytes(output)


def validate_xhtml_fragment(data):
    """Validate and summarize a serialized XHTML fragment manifest.

    Validation is intentionally independent of XML DTDs and network access.
    It rejects active content, external rendering resources, source ``data-*``
    attributes, undeclared internal links, duplicate identifiers and malformed
    page metadata.  Declared opaque cross-fragment link tokens and normal
    hyperlinks to HTTP(S), email and telephone targets remain allowed.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    manifest = json.loads(data)
    if manifest.get('format') != FRAGMENT_FORMAT:
        raise ValueError('Unknown XHTML fragment format')
    if manifest.get('version') != FRAGMENT_VERSION:
        raise ValueError('Unsupported XHTML fragment version')
    if manifest.get('css_unit') != 'px':
        raise ValueError('XHTML fragment coordinates must use CSS pixels')
    if not isinstance(manifest.get('stylesheet'), str):
        raise ValueError('XHTML fragment stylesheet is missing')
    if _UNSAFE_CSS_PATTERN.search(manifest['stylesheet']):
        raise ValueError('Unsafe content in XHTML fragment stylesheet')
    for match in _URL_PATTERN.finditer(manifest['stylesheet']):
        if not match.group(2).strip().lower().startswith('data:'):
            raise ValueError('External rendering resource in stylesheet')

    ids = set()
    hrefs = []
    summary = {
        'pages': 0, 'elements': 0, 'text_characters': 0,
        'images': 0, 'tables': 0}
    for expected_number, page in enumerate(manifest.get('pages', ()), 1):
        if page.get('number') != expected_number:
            raise ValueError('XHTML fragment pages are not sequential')
        if page.get('width', 0) <= 0 or page.get('height', 0) <= 0:
            raise ValueError('XHTML fragment page has invalid dimensions')
        try:
            root = ElementTree.fromstring(page['xhtml'])
        except (ElementTree.ParseError, KeyError) as exception:
            raise ValueError('Malformed XHTML page') from exception
        if root.tag != f'{_XHTML}section':
            raise ValueError('XHTML page root must be a namespaced section')
        if root.get('id') != page.get('id'):
            raise ValueError('XHTML page identifier does not match manifest')
        if root.get('class') != 'wp-page':
            raise ValueError('XHTML page root must use the generated page class')
        summary['pages'] += 1
        for element in root.iter():
            summary['elements'] += 1
            summary['text_characters'] += len(element.text or '')
            tag = _local_name(element.tag)
            if tag in _FORBIDDEN_ELEMENTS or tag == 'style':
                raise ValueError(f'Forbidden XHTML element: {tag}')
            if tag == 'img':
                summary['images'] += 1
            elif tag == 'table':
                summary['tables'] += 1
            identifier = element.get('id')
            if identifier:
                if identifier in ids:
                    raise ValueError(f'Duplicate XHTML identifier: {identifier}')
                ids.add(identifier)
            for name, value in element.attrib.items():
                local = _local_name(name)
                if local.startswith(('data-', 'on')):
                    raise ValueError(f'Forbidden XHTML attribute: {local}')
                if local == 'src':
                    lower = value.lower()
                    if not lower.startswith(tuple(
                            f'data:{mime};base64,' for mime in _ALLOWED_IMAGE_MIMES)):
                        raise ValueError('External or unsupported image resource')
                elif local == 'href':
                    if value.startswith('#'):
                        hrefs.append(value[1:])
                    elif urlparse(value).scheme.lower() not in _ALLOWED_LINK_SCHEMES:
                        raise ValueError('Unsafe XHTML hyperlink')
                elif local == 'style':
                    if _UNSAFE_CSS_PATTERN.search(value):
                        raise ValueError('Unsafe inline CSS')
                    for match in _URL_PATTERN.finditer(value):
                        if not match.group(2).strip().lower().startswith('data:'):
                            raise ValueError(
                                'External rendering resource in inline CSS')
    links = manifest.get('links', [])
    if not isinstance(links, list):
        raise ValueError('XHTML fragment links must be a list')
    link_tokens = set()
    link_sources = set()
    expected_prefix = f'{manifest.get("fragment_id", "")}-l'
    for link in links:
        if not isinstance(link, dict):
            raise ValueError('XHTML fragment link must be an object')
        token = link.get('token')
        source = link.get('source')
        if (
                not isinstance(token, str) or
                re.fullmatch(rf'{re.escape(expected_prefix)}[1-9]\d*', token)
                is None):
            raise ValueError('XHTML fragment link has an invalid opaque token')
        if not isinstance(source, str) or not source:
            raise ValueError('XHTML fragment link has an invalid source anchor')
        if token in link_tokens or source in link_sources:
            raise ValueError('Duplicate XHTML fragment link mapping')
        link_tokens.add(token)
        link_sources.add(source)
    if link_tokens & ids:
        raise ValueError('XHTML fragment link token collides with an element id')
    unresolved = sorted(set(hrefs) - ids - link_tokens)
    if unresolved:
        raise ValueError(f'Unresolved internal XHTML links: {unresolved!r}')
    unused_tokens = sorted(link_tokens - set(hrefs))
    if unused_tokens:
        raise ValueError(f'Unused XHTML fragment link tokens: {unused_tokens!r}')
    summary['cross_fragment_links'] = len(link_tokens)
    anchor_targets = [
        anchor.get('target') for anchor in manifest.get('anchors', ())]
    if None in anchor_targets or len(anchor_targets) != len(set(anchor_targets)):
        raise ValueError('Invalid or duplicate manifest anchor targets')
    missing_targets = sorted(set(anchor_targets) - ids)
    if missing_targets:
        raise ValueError(
            f'Manifest anchors without XHTML targets: {missing_targets!r}')
    if any(not isinstance(anchor.get('page'), int)
           for anchor in manifest.get('anchors', ())):
        raise ValueError('Manifest anchors must have a page number')
    if summary['pages'] != len(manifest.get('pages', ())):
        raise ValueError('XHTML fragment page count mismatch')
    return summary


__all__ = [
    'FRAGMENT_FORMAT', 'FRAGMENT_VERSION', 'export_xhtml_fragment',
    'validate_xhtml_fragment', 'write_xhtml_fragment']
