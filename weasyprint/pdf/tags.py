"""PDF tagging."""

from collections import defaultdict

import pydyf

from ..formatting_structure import boxes
from ..layout.absolute import AbsolutePlaceholder
from ..logger import LOGGER


def add_tags(pdf, document, page_streams):
    """Add tag tree to the document."""

    # Add root structure.
    content_mapping = pydyf.Dictionary({})
    pdf.add_object(content_mapping)
    # Name tree mapping structure element IDs (strings) to struct elements
    id_mapping = pydyf.Dictionary({})
    pdf.add_object(id_mapping)
    structure_root = pydyf.Dictionary({
        'Type': '/StructTreeRoot',
        'ParentTree': content_mapping.reference,
        'IDTree': id_mapping.reference,
    })
    pdf.add_object(structure_root)
    structure_document = pydyf.Dictionary({
        'Type': '/StructElem',
        'S': '/Document',
        'K': pydyf.Array(),
        'P': structure_root.reference,
    })
    pdf.add_object(structure_document)
    structure_root['K'] = pydyf.Array([structure_document.reference])
    pdf.catalog['StructTreeRoot'] = structure_root.reference

    # Map content.
    content_mapping['Nums'] = pydyf.Array()
    id_mapping['Names'] = pydyf.Array()
    links = []
    for page_number, (page, stream) in enumerate(zip(document.pages, page_streams)):
        tags = stream._tags
        page_box = page._page_box

        # Prepare array for this page’s MCID-to-StructElem mapping.
        content_mapping['Nums'].append(page_number)
        content_mapping['Nums'].append(pydyf.Array())
        page_nums = {}

        # Map page box content.
        # Collect mapping between table cell boxes and their struct elements
        cell_elements = {}
        elements = _build_box_tree(
            page_box, structure_document, pdf, page_number,
            page_nums, links, tags, id_mapping, cell_elements)
        for element in elements:
            structure_document['K'].append(element.reference)
        assert not tags

        # Flatten page-local nums into global mapping.
        sorted_refs = [ref for _, ref in sorted(page_nums.items())]
        content_mapping['Nums'][-1].extend(sorted_refs)

    # Add annotations for links.
    for i, (link_reference, annotation) in enumerate(links, start=len(document.pages)):
        content_mapping['Nums'].append(i)
        content_mapping['Nums'].append(link_reference)
        annotation['StructParent'] = i

    # Add required metadata.
    pdf.catalog['ViewerPreferences'] = pydyf.Dictionary({'DisplayDocTitle': 'true'})
    pdf.catalog['MarkInfo'] = pydyf.Dictionary({'Marked': 'true'})
    if 'Lang' not in pdf.catalog:
        LOGGER.error('Missing required "lang" attribute at the root of the document')
        pdf.catalog['Lang'] = pydyf.String()


def _get_pdf_tag(tag):
    """Get PDF tag corresponding to HTML tag."""
    if tag is None:
        return 'NonStruct'
    elif tag == 'div':
        return 'Div'
    elif tag.split(':')[0] == 'a':
        # Links and link pseudo elements create link annotations.
        return 'Link'
    elif tag == 'span':
        return 'Span'
    elif tag == 'main':
        return 'Part'
    elif tag == 'article':
        return 'Art'
    elif tag == 'section':
        return 'Sect'
    elif tag == 'blockquote':
        return 'BlockQuote'
    elif tag == 'p':
        return 'P'
    elif tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
        return tag.upper()
    elif tag in ('dl', 'ul', 'ol'):
        return 'L'
    elif tag in ('li', 'dt', 'dd'):
        # TODO: dt should be different.
        return 'LI'
    elif tag == 'li::marker':
        return 'Lbl'
    elif tag == 'table':
        return 'Table'
    elif tag in ('tr', 'th', 'td'):
        return tag.upper()
    elif tag in ('thead', 'tbody', 'tfoot'):
        return tag[:2].upper() + tag[2:]
    elif tag == 'img':
        return 'Figure'
    elif tag in ('caption', 'figcaption'):
        return 'Caption'
    else:
        return 'NonStruct'


def _build_box_tree(
    box, parent, pdf, page_number, nums, links, tags, id_tree, cell_elements
):
    """Recursively build tag tree for given box and yield children."""

    # Special case for absolute elements.
    if isinstance(box, AbsolutePlaceholder):
        box = box._box

    element_tag = None if box.element is None else box.element_tag
    # Skip aria-hidden subtrees entirely in the structure tree.
    if getattr(box, 'aria_hidden', False):
        return
    tag = _get_pdf_tag(element_tag)

    # Special case for html, body, page boxes and margin boxes.
    if element_tag in ('html', 'body') or isinstance(box, boxes.PageBox):
        # Avoid generate page, html and body boxes as a semantic node, yield children.
        if isinstance(box, boxes.ParentBox) and not isinstance(box, boxes.LineBox):
            for child in box.children:
                yield from _build_box_tree(
                    child, parent, pdf, page_number, nums, links, tags, id_tree, cell_elements)
            return
    elif isinstance(box, boxes.MarginBox):
        # Build tree for margin boxes but don’t link it to main tree. It ensures that
        # marked content is mapped in document and removed from list. It could be
        # included in tree as Artifact, but that’s only allowed in PDF 2.0.
        for child in box.children:
            tuple(_build_box_tree(child, parent, pdf, page_number, nums, links, tags, id_tree, cell_elements))
        return

    # Create box element.
    if tag == 'LI':
        anonymous_list_element = parent['S'] == '/LI'
        anonymous_li_child = parent['S'] == '/LBody'
        dl_item = box.element_tag in ('dt', 'dd')
        no_bullet_li = box.element_tag == 'li' and (
            'list-item' not in box.style['display'] or
            box.style['list_style_type'] == 'none')
        if anonymous_list_element:
            # Store as list item body.
            tag = 'LBody'
        elif anonymous_li_child:
            # Store as non struct list item body child.
            tag = 'NonStruct'
        elif dl_item or no_bullet_li:
            # Wrap in list item.
            tag = 'LBody'
            parent = pydyf.Dictionary({
                'Type': '/StructElem',
                'S': '/LI',
                'K': pydyf.Array([]),
                'Pg': pdf.page_references[page_number],
                'P': parent.reference,
            })
            pdf.add_object(parent)
            children = _build_box_tree(box, parent, pdf, page_number, nums, links, tags, id_tree, cell_elements)
            for child in children:
                parent['K'].append(child.reference)
            yield parent
            return

    element = pydyf.Dictionary({
        'Type': '/StructElem',
        'S': f'/{tag}',
        'K': pydyf.Array([]),
        'Pg': pdf.page_references[page_number],
        'P': parent.reference,
    })
    pdf.add_object(element)

    # Handle special cases.
    if tag == 'Figure':
        # Add extra data for images.
        x1, y1 = box.content_box_x(), box.content_box_y()
        x2, y2 = x1 + box.width, y1 + box.height
        element['A'] = pydyf.Dictionary({
            'O': '/Layout',
            'BBox': pydyf.Array((x1, y1, x2, y2)),
        })
        if alt := box.element.attrib.get('alt'):
            element['Alt'] = pydyf.String(alt)
        else:
            source = box.element.attrib.get('src', 'unknown')
            LOGGER.error(f'Image "{source}" has no required alt description')
    elif tag == 'Table':
        # Use wrapped table as tagged box, and put captions in it.
        wrapper, table = box, box.get_wrapped_table()
        box = table.copy_with_children([])
        for child in wrapper.children:
            box.children.extend(child.children if child is table else [child])
    elif tag == 'TH':
        # Set identifier for table headers to reference them in cells
        # and register it in the document ID tree.
        # Prefer the HTML id if available to support the HTML headers attribute.
        html_id = None
        if box.element is not None:
            html_id = box.element.attrib.get('id')
        th_id = pydyf.String(html_id if html_id else id(box))
        element['ID'] = th_id
        # Add Scope attribute when available from HTML.
        scope_attr = None
        if box.element is not None:
            scope_attr = box.element.attrib.get('scope')
        if scope_attr in ('row', 'col', 'rowgroup', 'colgroup', 'both'):
            # Map HTML scope to PDF Table Scope. RowGroup/ColGroup map to Row/Column.
            if scope_attr == 'both':
                scope_value = '/Both'
            else:
                scope_value = '/Row' if scope_attr in ('row', 'rowgroup') else '/Column'
            element['A'] = pydyf.Dictionary({
                'O': '/Table',
                'Scope': scope_value,
            })
        # Register the ID so that Headers can be resolved algorithmically.
        id_tree['Names'].append(th_id)
        id_tree['Names'].append(element.reference)
        cell_elements[box] = element
    elif tag == 'TD':
        # Store table cell element to map it to headers later.
        # Do not mutate the box; keep a side mapping instead.
        cell_elements[box] = element

    # Include link annotations.
    if box.link_annotation:
        annotation = box.link_annotation
        object_reference = pydyf.Dictionary({
            'Type': '/OBJR',
            'Obj': annotation.reference,
            'Pg': pdf.page_references[page_number],
        })
        pdf.add_object(object_reference)
        links.append((element.reference, annotation))
        element['K'].append(object_reference.reference)

    if isinstance(box, boxes.ParentBox):
        # Build tree for box children.
        for child in box.children:
            children = child.children if isinstance(child, boxes.LineBox) else [child]
            for child in children:
                if isinstance(child, boxes.TextBox):
                    # Add marked element from the stream if present.
                    if child not in tags:
                        continue
                    kid = tags.pop(child)
                    assert kid['mcid'] not in nums
                    if tag == 'Link':
                        # Associate MCID directly with link reference.
                        element['K'].append(kid['mcid'])
                        nums[kid['mcid']] = element.reference
                    else:
                        kid_element = pydyf.Dictionary({
                            'Type': '/StructElem',
                            'S': f'/{kid["tag"]}',
                            'K': pydyf.Array([kid['mcid']]),
                            'Pg': pdf.page_references[page_number],
                            'P': element.reference,
                        })
                        pdf.add_object(kid_element)
                        element['K'].append(kid_element.reference)
                        nums[kid['mcid']] = kid_element.reference
                else:
                    # Recursively build tree for child.
                    if child.element_tag in ('ul', 'ol') and element['S'] == '/LI':
                        # In PDFs, nested lists are linked to the parent list, but in
                        # HTML, nested lists are linked to a parent’s list item.
                        child_parent = parent
                    else:
                        child_parent = element
                    child_elements = _build_box_tree(
                        child, child_parent, pdf, page_number, nums, links, tags, id_tree, cell_elements)

                    # Check if it is already been referenced before.
                    for child_element in child_elements:
                        child_parent['K'].append(child_element.reference)

    else:
        # Add replaced box.
        assert isinstance(box, boxes.ReplacedBox)
        if box in tags:
            kid = tags.pop(box)
            element['K'].append(kid['mcid'])
            assert kid['mcid'] not in nums
            nums[kid['mcid']] = element.reference

    # Link table cells to related headers with spans and groups.
    if tag == 'Table':
        # Alias to the actual laid-out table for column group data.
        # 'table' is defined above in the Table special-case block.

        # Build rows with their row group, and compute grid width.
        rows = []
        row_groups = []
        ncols = 0
        for group in table.children:
            if not isinstance(group, boxes.TableRowGroupBox):
                continue
            group_index = len(row_groups)
            row_groups.append(group)
            for row in group.children:
                rows.append((group_index, group, row))
                for cell in row.children:
                    if cell.element is None:
                        continue
                    ncols = max(ncols, getattr(cell, 'grid_x', 0) + getattr(cell, 'colspan', 1))

        # Map each column index to a column group index when defined.
        colgroup_by_col = [None] * ncols
        for colg_index, colg in enumerate(getattr(table, 'column_groups', ())):
            start = getattr(colg, 'grid_x', 0)
            if colg.children:
                span = len(colg.children)
            else:
                span = colg.span
            for x in range(start, min(start + span, ncols)):
                colgroup_by_col[x] = colg_index

        # Build a cell grid: for each row and column, which cell covers it.
        grid = [[None for _ in range(ncols)] for _ in range(len(rows))]
        for r, (_, _, row) in enumerate(rows):
            for cell in row.children:
                if cell.element is None:
                    continue
                x0 = getattr(cell, 'grid_x', 0)
                x1 = x0 + getattr(cell, 'colspan', 1)
                for x in range(x0, min(x1, ncols)):
                    grid[r][x] = cell

        # Helper to compute TH id string used in PDF Headers mapping.
        def th_id_string(th_cell):
            html_id = None
            if th_cell.element is not None:
                html_id = th_cell.element.attrib.get('id')
            return pydyf.String(html_id if html_id else id(th_cell))

        # Initialize headers mapping for each TD.
        td_headers = defaultdict(list)

        # Compute default scope heuristics.
        def inferred_scope(th_cell, group_obj, row_obj):
            if th_cell.element is not None:
                scope_attr = th_cell.element.attrib.get('scope')
                if scope_attr in ('row', 'col', 'rowgroup', 'colgroup', 'both'):
                    return scope_attr
            # Heuristics when scope is absent
            if getattr(group_obj, 'is_header', False):
                return 'col'
            # First originating cell in row → row header
            first_cell = None
            for c in row_obj.children:
                if c.element is not None:
                    first_cell = c
                    break
            if first_cell is th_cell:
                return 'row'
            return 'col'

        # Apply TH headers based on scope.
        for r, (g_index, group, row) in enumerate(rows):
            for th_cell in row.children:
                if th_cell.element is None or th_cell.element_tag != 'th':
                    continue
                header_id = th_id_string(th_cell)
                x0 = getattr(th_cell, 'grid_x', 0)
                x1 = x0 + getattr(th_cell, 'colspan', 1)
                y0 = r
                y1 = r + max(getattr(th_cell, 'rowspan', 1), 1)
                scope_attr = inferred_scope(th_cell, group, row)

                def add_row_headers(row_index):
                    # Add header to all TDs in the row, excluding the TH itself.
                    if 0 <= row_index < len(rows):
                        for x in range(ncols):
                            cell = grid[row_index][x]
                            if cell is None or cell.element is None:
                                continue
                            if cell.element_tag != 'td':
                                continue
                            # Skip cells entirely under the TH rectangle
                            if y0 <= row_index < y1 and x0 <= x < x1:
                                continue
                            if header_id not in td_headers[cell]:
                                td_headers[cell].append(header_id)

                def add_col_headers(col_index, row_range=None):
                    # Add header to all TDs in the column within row_range.
                    rows_iter = range(len(rows)) if row_range is None else row_range
                    if 0 <= col_index < ncols:
                        for ry in rows_iter:
                            cell = grid[ry][col_index]
                            if cell is None or cell.element is None:
                                continue
                            if cell.element_tag != 'td':
                                continue
                            if header_id not in td_headers[cell]:
                                td_headers[cell].append(header_id)

                if scope_attr == 'row':
                    for ry in range(y0, min(y1, len(rows))):
                        add_row_headers(ry)
                elif scope_attr == 'col':
                    for x in range(x0, min(x1, ncols)):
                        add_col_headers(x)
                elif scope_attr == 'both':
                    for ry in range(y0, min(y1, len(rows))):
                        add_row_headers(ry)
                    for x in range(x0, min(x1, ncols)):
                        add_col_headers(x)
                elif scope_attr == 'rowgroup':
                    # Add as row header for each row in the same row group
                    # as the TH cell.
                    # Find the contiguous range of rows belonging to this group.
                    for rr, (gg_index, _, _) in enumerate(rows):
                        if gg_index == g_index:
                            add_row_headers(rr)
                elif scope_attr == 'colgroup':
                    # Determine colgroup(s) covered by the TH and add
                    # as column headers for all columns in these groups.
                    colgroups = set()
                    for x in range(x0, min(x1, ncols)):
                        cg = colgroup_by_col[x]
                        if cg is not None:
                            colgroups.add(cg)
                    if not colgroups:
                        # Fallback to columns spanned if no colgroup exists
                        for x in range(x0, min(x1, ncols)):
                            add_col_headers(x)
                    else:
                        for x, cg in enumerate(colgroup_by_col):
                            if cg in colgroups:
                                add_col_headers(x)
                else:
                    # Unknown scope value: default to column
                    for x in range(x0, min(x1, ncols)):
                        add_col_headers(x)

        # After header inference, apply explicit HTML headers attribute
        # when present on TD cells (overrides inference).
        for r, (_, _, row) in enumerate(rows):
            for cell in row.children:
                if cell.element is None or cell.element_tag != 'td':
                    continue
                elem = cell_elements.get(cell)
                if elem is None:
                    continue
                if cell.element is not None and 'headers' in cell.element.attrib:
                    explicit_ids = [
                        pydyf.String(tok) for tok in cell.element.attrib.get('headers', '').split()
                    ]
                    if explicit_ids:
                        elem['A'] = pydyf.Dictionary({
                            'O': '/Table',
                            'Headers': pydyf.Array(explicit_ids),
                        })
                        continue
                # Fallback to inferred headers if any
                if cell in td_headers and td_headers[cell]:
                    elem['A'] = pydyf.Dictionary({
                        'O': '/Table',
                        'Headers': pydyf.Array(td_headers[cell]),
                    })

    yield element
