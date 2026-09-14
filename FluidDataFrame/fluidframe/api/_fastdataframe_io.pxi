# _fastdataframe_io.pxi — I/O and display (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)


cdef list _col_to_list(BaseColumn col):
    """
    【核心加速机制】：一次性探测列的 shape（1D 还是 2D），成 Python list 一次性返回。
    """
    cdef bint is2d
    cdef Py_ssize_t c, r
    is2d, c, r = _col_shape_probe(col)

    if r == 0:
        return []

    cdef Py_ssize_t i, j
    cdef list res = [None] * r
    cdef list row_vals

    # 极速批处理 Dump
    if is2d and c > 1:
        for i in range(r):
            row_vals = [None] * c
            for j in range(c):
                row_vals[j] = col[j * r + i]
            res[i] = row_vals
    else:
        for i in range(r):
            res[i] = col[i]
    return res


cdef _to_string_impl(FastDataFrame self, Py_ssize_t max_rows=80):
    cdef Py_ssize_t n = self.df_length
    cdef list row_indices = []
    cdef Py_ssize_t half

    if n <= max_rows:
        row_indices = list(range(n))
    else:
        half = max_rows // 2
        row_indices = list(range(half)) + [-1] + list(range(n - half, n))

    names, types = self._get_header_strings()
    cdef int num_display_cols = len(names)

    cdef list table_data = []
    cdef Py_ssize_t r
    for r in row_indices:
        if r == -1:
            table_data.append(["..."] * num_display_cols)
        else:
            table_data.append(self._get_row_strings(r))

    cdef list col_widths = [0] * num_display_cols
    cdef int c, max_w, current_w
    for c in range(num_display_cols):
        max_w = self._display_width(names[c])
        current_w = self._display_width(types[c])
        if current_w > max_w: max_w = current_w
        for row in table_data:
            current_w = self._display_width(row[c])
            if current_w > max_w: max_w = current_w
        col_widths[c] = max_w + 2

    cdef list lines = []
    lines.append(f"shape: ({self.df_length}, {len(self._columns)})")

    if num_display_cols > 0:
        def make_border(left, mid, right, fill):
            return left + mid.join([fill * col_widths[k] for k in range(num_display_cols)]) + right

        lines.append(make_border("┌", "┬", "┐", "─"))

        def format_cell(text, width, align_right):
            dw = self._display_width(text)
            padding = width - dw
            return (" " * padding + text) if align_right else (text + " " * padding)

        row_str = "│"
        for c in range(num_display_cols):
            row_str += " " + format_cell(names[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
        lines.append(row_str)

        row_str = "│"
        for c in range(num_display_cols):
            row_str += " " + format_cell(types[c], col_widths[c] - 2, False) + " " + ("│" if c == num_display_cols - 1 else "┆")
        lines.append(row_str)

        lines.append(make_border("╞", "╪", "╡", "═"))

        for row_idx, row in enumerate(table_data):
            row_str = "│"
            for c in range(num_display_cols):
                val = row[c]
                is_num = val != "..." and types[c] in ("[i64]", "[f64]", "[range]", "[multi]")
                row_str += " " + format_cell(val, col_widths[c] - 2, is_num) + " " + ("│" if c == num_display_cols - 1 else "┆")
            lines.append(row_str)

        lines.append(make_border("└", "┴", "┘", "─"))
    else:
        lines.append("Empty DataFrame")

    return "\n".join(lines)

cdef _to_pickle_impl(FastDataFrame self, filename):
    import pickle
    cdef dict data = {'df_length': self.df_length, 'columns': {}}
    cdef bint is2d
    cdef Py_ssize_t c, r, i
    cdef Py_ssize_t total_cells
    cdef int dtype
    cdef list values
    cdef int32_t* dptr32
    cdef int64_t* tptr64
    cdef BaseColumn col_obj

    for name, col_obj in self._columns.items():
        is2d, c, r = _col_shape_probe(col_obj)
        dtype = col_obj.dtype
        if is2d and c > 1:
            total_cells = r * c
            if dtype == TYPE_DATE:
                dptr32 = <int32_t*>(<DateColumn>col_obj).data_c
                values = [dptr32[i] for i in range(total_cells)]
            elif dtype == TYPE_DATETIME:
                tptr64 = <int64_t*>(<DateTimeColumn>col_obj).data_c
                values = [tptr64[i] for i in range(total_cells)]
            elif dtype == TYPE_TIME:
                dptr32 = <int32_t*>(<TimeColumn>col_obj).data_c
                values = [dptr32[i] for i in range(total_cells)]
            else:
                values = [col_obj[i] for i in range(total_cells)]
            data['columns'][name] = (self._type_name(col_obj), values, r, c)
        else:
            values = [col_obj[i] for i in range(r)]
            data['columns'][name] = (self._type_name(col_obj), values)

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

cdef _to_csv_impl(FastDataFrame self, filename, sep=','):
    """
    导出为 CSV 文件。
    优化点：批量列数据提取，摆脱嵌套循环中的重复类型查询。
    """
    cdef list col_names = list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t i, j

    cdef size_t buf_pos = 0
    cdef size_t write_res
    cdef char sep_c = <char>ord(sep) if len(sep) == 1 else <char>ord(',')
    cdef char line_buf[65536]
    cdef size_t buf_cap = sizeof(line_buf)
    cdef bytes b_field

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]

    with open(filename, 'wb') as f:
        for j in range(num_cols):
            b_field = col_names[j].encode('utf-8')
            if j > 0:
                line_buf[buf_pos] = sep_c
                buf_pos += 1

            write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
            if write_res == <size_t>-1:
                f.write(line_buf[:buf_pos])
                buf_pos = 0
                write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
            buf_pos = write_res

        line_buf[buf_pos] = 10
        buf_pos += 1
        f.write(line_buf[:buf_pos])


        for i in range(n):
            buf_pos = 0
            for j in range(num_cols):
                b_field = str(cols_data[j][i]).encode('utf-8')

                if j > 0:
                    line_buf[buf_pos] = sep_c
                    buf_pos += 1
                write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                if write_res == <size_t>-1:
                    f.write(line_buf[:buf_pos])
                    buf_pos = 0
                    write_res = _csv_write_field(line_buf, buf_cap, buf_pos, <const char*>b_field, len(b_field), sep_c)
                buf_pos = write_res

            line_buf[buf_pos] = 10
            buf_pos += 1
            f.write(line_buf[:buf_pos])

cdef _to_numpy_impl(FastDataFrame self):
    """
    导出为二维数组 (List of Lists)。
    """
    cdef Py_ssize_t n = self.df_length
    cdef list col_names = list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t i, j

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]
    cdef list result = [None] * n
    cdef list row

    for i in range(n):
        row = [None] * num_cols
        for j in range(num_cols):
            row[j] = cols_data[j][i]
        result[i] = row
    return result

cdef _info_impl(FastDataFrame self):
    cdef Py_ssize_t total_cols = len(self._columns)
    cdef Py_ssize_t mem = _memory_usage_impl(self)
    cdef char bytes_buf[16]
    cdef list lines = []
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t col_bytes, col_rows
    cdef bint is2d
    cdef Py_ssize_t c_probe, r_probe
    cdef str type_name, name
    cdef object col

    lines.append("<class 'FastDataFrame'>")
    lines.append(f"RangeIndex: {self.df_length} entries")
    lines.append(f"Data columns (total {total_cols} columns):")
    lines.append(" #   Column    Non-Null  Dtype     Mem Usage")
    lines.append("---  ------    --------  ----     ---------")

    for name, col in self._columns.items():
        type_name = self._type_name(col)
        col_bytes = col.get_size()
        is2d, c_probe, r_probe = _col_shape_probe(col)

        _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), col_bytes)
        lines.append(f" {idx}   {name:<10}{r_probe:<10}{type_name:<10}{bytes_buf.decode('ascii')}")
        idx += 1

    _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), mem)
    lines.append(f"dtypes: {_dtype_summary_impl(self)}")
    lines.append(f"memory usage: {bytes_buf.decode('ascii')}")
    return '\n'.join(lines)

cdef str _dtype_summary_impl(FastDataFrame self):
    cdef dict counts = {}
    cdef str name, tname
    for name, col in self._columns.items():
        tname = self._type_name(col)
        counts[tname] = counts.get(tname, 0) + 1
    return ', '.join(f'{v} {k}' for k, v in counts.items())

cdef _memory_usage_impl(FastDataFrame self):
    cdef Py_ssize_t total = 0
    cdef BaseColumn col
    for col in self._columns.values():
        total += col.get_size()
    return total

cdef _to_dict_impl(FastDataFrame self):
    """导出为字段字典 (Columnar Dictionary)"""
    cdef dict result = {}
    cdef str name
    cdef BaseColumn col

    for name, col in self._columns.items():
        result[name] = _col_to_list(col)
    return result

cdef _to_json_impl(FastDataFrame self, bint orient_records=True):
    """导出为 JSON"""
    import json

    cdef dict columns_dict = _to_dict_impl(self)

    if not orient_records:
        return json.dumps(columns_dict)

    # ---------------------------------------------------------
    # Orient Records 模式（行翻转矩阵）
    # ---------------------------------------------------------
    cdef Py_ssize_t n = self.df_length
    cdef list records = [None] * n

    cdef list col_names = list(columns_dict.keys())
    cdef Py_ssize_t num_cols = len(col_names)

    cdef list cols_data = list(columns_dict.values())
    cdef dict row_dict
    cdef Py_ssize_t i, j

    for i in range(n):
        row_dict = {}
        for j in range(num_cols):
            row_dict[col_names[j]] = cols_data[j][i]
        records[i] = row_dict
    return json.dumps(records)


cdef str _html_escape(object val):
    """HTML 特殊字符转义（合并进 to_html 内调用，避免额外 Cython 开销）。"""
    cdef str s = str(val)
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
             .replace('"', '&quot;').replace("'", '&#39;'))


cdef str _to_html_impl(FastDataFrame self, Py_ssize_t max_rows=80):
    """导出为 HTML 表格字符串（<table class="dataframe">）。"""
    cdef Py_ssize_t n = self.df_length
    cdef list row_indices = []
    cdef Py_ssize_t half, r

    if n <= max_rows:
        row_indices = list(range(n))
    else:
        half = max_rows // 2
        row_indices = list(range(half)) + [-1] + list(range(n - half, n))

    cdef list names = self._get_header_strings()[0]
    cdef Py_ssize_t num_display_cols = len(names)

    cdef list rows = []
    for r in row_indices:
        rows.append(["..."] * num_display_cols if r == -1 else self._get_row_strings(r))

    cdef list out = ['<table class="dataframe">\n  <thead>', '    <tr>']
    for c in range(num_display_cols):
        out.append(f"      <th>{_html_escape(names[c])}</th>")
    out.append('    </tr>\n    </thead>')
    out.append('  <tbody>')
    for row in rows:
        out.append('    <tr>')
        for c in range(num_display_cols):
            out.append(f"      <td>{_html_escape(row[c])}</td>")
        out.append('    </tr>')
    out.append('  </tbody>')
    out.append('</table>')
    return '\n'.join(out)


cdef _to_excel_impl(FastDataFrame self, object filename, str sheet_name='Sheet1'):
    """导出为 xlsx 文件（openpyxl 懒加载；未安装时抛 ImportError 提示）。"""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("to_excel() requires 'openpyxl'. Install with: pip install openpyxl")

    cdef list col_names = list(self._columns.keys())
    cdef Py_ssize_t num_cols = len(col_names)
    cdef Py_ssize_t n = self.df_length
    cdef Py_ssize_t i, j

    cdef list cols_data = [_col_to_list(self._columns[name]) for name in col_names]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append([c for c in col_names])
    for i in range(n):
        row = [None] * num_cols
        for j in range(num_cols):
            v = cols_data[j][i]
            row[j] = v if v is not None else ''
        ws.append(row)
    wb.save(filename)
    return filename
