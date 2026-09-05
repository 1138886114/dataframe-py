# _fastdataframe_io.pxi — I/O and display (module-level cdef)
# Included at module level in fastdataframe.pyx (after class definition)

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
    lines.append(f"shape: ({self.df_length}, {len(self.columns)})")

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
    for name, col_obj in self.columns.items():
        col_len = col_obj.length
        values = [col_obj[i] for i in range(col_len)]
        data['columns'][name] = (self._type_name(col_obj), values)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

cdef _to_csv_impl(FastDataFrame self, filename, sep=','):
    cdef list col_names = list(self.columns.keys())
    cdef list col_objs = [self.columns[name] for name in col_names]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t num_cols = len(col_names)
    cdef size_t buf_pos = 0
    cdef size_t write_res
    cdef char sep_c = <char>ord(sep) if len(sep) == 1 else <char>ord(',')
    cdef char line_buf[65536]
    cdef size_t buf_cap = sizeof(line_buf)
    cdef bytes b_field

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

        for i in range(self.df_length):
            buf_pos = 0
            for j in range(num_cols):
                b_field = str(col_objs[j][i]).encode('utf-8')
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
    cdef list result = []
    cdef list row
    cdef Py_ssize_t i, j
    cdef list col_names = list(self.columns.keys())
    cdef list col_objs = [self.columns[name] for name in col_names]
    cdef Py_ssize_t num_cols = len(col_names)
    for i in range(self.df_length):
        row = []
        for j in range(num_cols):
            row.append(col_objs[j][i])
        result.append(row)
    return result

cdef _info_impl(FastDataFrame self):
    cdef Py_ssize_t total_cols = len(self.columns)
    cdef Py_ssize_t mem = _memory_usage_impl(self)
    cdef char bytes_buf[16]
    cdef list lines = []
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t col_bytes
    cdef str type_name, name
    cdef object col

    lines.append("<class 'FastDataFrame'>")
    lines.append(f"RangeIndex: {self.df_length} entries")
    lines.append(f"Data columns (total {total_cols} columns):")
    lines.append(" #   Column    Non-Null  Dtype     Mem Usage")
    lines.append("---  ------    --------  ----     ---------")

    for name, col in self.columns.items():
        type_name = self._type_name(col)
        col_bytes = col.get_size()
        _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), col_bytes)
        lines.append(f" {idx}   {name:<10}{col.length:<10}{type_name:<10}{bytes_buf.decode('ascii')}")
        idx += 1

    _c_fmt_bytes(bytes_buf, sizeof(bytes_buf), mem)
    lines.append(f"dtypes: {_dtype_summary_impl(self)}")
    lines.append(f"memory usage: {bytes_buf.decode('ascii')}")
    return '\n'.join(lines)

cdef str _dtype_summary_impl(FastDataFrame self):
    cdef dict counts = {}
    cdef str name, tname
    for name, col in self.columns.items():
        tname = self._type_name(col)
        counts[tname] = counts.get(tname, 0) + 1
    return ', '.join(f'{v} {k}' for k, v in counts.items())

cdef _memory_usage_impl(FastDataFrame self):
    cdef Py_ssize_t total = 0
    cdef BaseColumn col
    for col in self.columns.values():
        total += col.get_size()
    return total
