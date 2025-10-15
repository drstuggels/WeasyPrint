"""Tests for PDF table tagging: headers associations and scope."""

import re

from .testing_utils import FakeHTML, assert_no_logs


@assert_no_logs
def test_pdf_table_headers_scope_and_ids():
    html = '''
      <html lang="en">
      <head><meta charset="utf-8"></head>
      <body>
      <table>
        <colgroup span="2"></colgroup>
        <colgroup><col><col></colgroup>
        <thead>
          <tr>
            <th id="th_rowhdr" scope="col">Row</th>
            <th id="th_col1" scope="col">C1</th>
            <th id="th_both" scope="both">C2/Row</th>
            <th id="th_cg" scope="colgroup">Group</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th id="row1" scope="row">Row 1</th>
            <td id="td1">A</td>
            <td id="td2" headers="row1 th_cg">B</td>
            <td id="td3" rowspan="2">C</td>
          </tr>
          <tr>
            <th id="row2">Row 2</th>
            <td id="td4" colspan="2">D</td>
          </tr>
        </tbody>
      </table>
      </body>
      </html>
    '''

    pdf = FakeHTML(string=html).write_pdf(pdf_variant='pdf/ua-1')

    # Struct tree and ID tree exist
    assert b'/StructTreeRoot' in pdf
    assert b'/IDTree' in pdf

    # Scope attributes for THs appear
    assert b'/Scope /Row' in pdf
    assert b'/Scope /Column' in pdf
    assert b'/Scope /Both' in pdf

    # Named IDs for headers appear in the file
    for th_id in (b'row1', b'th_col1', b'th_cg', b'th_both'):
        assert b'(' + th_id + b')' in pdf

    # Collect all Table Headers arrays
    arrays = re.findall(br'/Headers\s*\[(.*?)\]', pdf, flags=re.S)
    def array_ids(arr_bytes):
        return set(re.findall(br'\(([^)]*)\)', arr_bytes))
    headers_sets = [array_ids(arr) for arr in arrays]

    # Explicit headers on td2 override inference: exactly row1 + th_cg
    assert any(ids == {b'row1', b'th_cg'} for ids in headers_sets)

    # Inferred headers for td1 include row1 and th_col1
    assert any({b'row1', b'th_col1'}.issubset(ids) for ids in headers_sets)

