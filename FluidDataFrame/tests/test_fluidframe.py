import sys
sys.path.insert(0, '.')
from fluidframe.core.fastdataframe import FastDataFrame
from fluidframe._schema_types import Char256, Date

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  FAIL: {name}')

def expect_exception(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except exc_type:
        return True
    except Exception:
        return False

def _make_df():
    df = FastDataFrame({'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
    rows = [
        (1, 88.5, 'Alice', 'senior', 20230115), (2, 92.0, 'Bob', 'mid', 20230620),
        (3, 75.3, 'Charlie', 'senior', 20221201), (4, 60.0, 'Diana', 'low', 20240229),
        (5, 95.1, 'Eve', 'senior', 20230115), (6, 82.7, 'Frank', 'mid', 20230310),
        (7, 70.0, 'Grace', 'low', 20220615), (8, 88.5, 'Heidi', 'senior', 20231225),
        (9, 99.9, 'Ivan', 'mid', 20240101), (10, 55.0, 'Judy', 'low', 20230620),
    ]
    for r in rows:
        df.append_row(id=r[0], score=r[1], name=r[2], tag=r[3], dt=r[4])
    return df


class TestReadCsv:
    def test_csv(self):
        import tempfile, os
        df = _make_df()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            fname = f.name
        try:
            df.to_csv(fname)
            from fluidframe.core.fluidframe import read_csv
            df2 = read_csv(fname, schema={'id': int, 'score': float, 'name': str, 'tag': Char256, 'dt': Date})
            check('csv shape', df2.shape == (10, 5))
            check('csv id[0]', df2['id'][0] == 1)
            check('csv name[4]', df2['name'][4] == 'Eve')
        finally:
            os.unlink(fname)

    def test_read_csv_auto(self):
        import tempfile, os
        content = "id,score,name,tag,dt\n1,88.5,Alice,senior,20230115\n2,92.0,Bob,mid,20230620\n"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(content); fname = f.name
        try:
            from fluidframe.core.fluidframe import read_csv
            df = read_csv(fname, schema={'id': int, 'score': float, 'name': str, 'tag': str, 'dt': int})
            check('shape', df.shape == (2, 5))
            check('id[0]', df['id'][0] == 1)
        finally:
            os.unlink(fname)


class TestPickle:
    def test_pickle(self):
        import tempfile, os
        df = _make_df()
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            fname = f.name
        try:
            df.to_pickle(fname)
            from fluidframe.core.fluidframe import from_pickle
            df2 = from_pickle(fname)
            check('shape', df2.shape == (10, 5))
            check('id[3]', df2['id'][3] == 4)
        finally:
            os.unlink(fname)


class TestTsv:
    def test_tsv(self):
        import tempfile, os
        df = _make_df()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False, newline='') as f:
            fname = f.name
        try:
            df.to_tsv(fname)
            check('exists', os.path.exists(fname))
        finally:
            os.unlink(fname)


class TestConcat:
    def test_concat_empty(self):
        from fluidframe.core.fluidframe import concat_outer
        check('empty', concat_outer([]).shape == (0, 0))

    def test_concat_two(self):
        from fluidframe.core.fluidframe import concat_outer
        df1 = _make_df().head(5)
        df2 = _make_df().tail(5)
        r = concat_outer([df1, df2])
        check('shape', r.shape == (10, 6))
        check('has index', 'index' in r.columns)

    def test_concat_basic(self):
        from fluidframe.core.fluidframe import concat
        df1 = FastDataFrame({'A': int, 'B': int})
        df1.append_row(A=1, B=10); df1.append_row(A=2, B=20)
        df2 = FastDataFrame({'A': int, 'B': int})
        df2.append_row(A=3, B=30)
        result = concat([df1, df2])
        check('len', result.df_length == 3)
        check('orig unchanged', df1.df_length == 2)

    def test_concat_ignore_index(self):
        from fluidframe.core.fluidframe import concat
        df1 = FastDataFrame({'A': int}); df1.append_row(A=1)
        df2 = FastDataFrame({'A': int}); df2.append_row(A=2)
        result = concat([df1, df2], ignore_index=True)
        check('idx', result.index.values() == [0, 1])


class TestMerge:
    def test_inner(self):
        df1 = FastDataFrame({'id': int, 'val1': float})
        df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
        df2 = FastDataFrame({'id': int, 'val2': float})
        df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
        check('inner', df1.merge(df2, on='id', how='inner').shape == (1, 3))

    def test_left(self):
        df1 = FastDataFrame({'id': int, 'val1': float})
        df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
        df2 = FastDataFrame({'id': int, 'val2': float})
        df2.append_row(id=2, val2=200.0)
        check('left', df1.merge(df2, on='id', how='left').shape == (2, 3))

    def test_outer(self):
        df1 = FastDataFrame({'id': int, 'val1': float})
        df1.append_row(id=1, val1=10.0); df1.append_row(id=2, val1=20.0)
        df2 = FastDataFrame({'id': int, 'val2': float})
        df2.append_row(id=2, val2=200.0); df2.append_row(id=3, val2=300.0)
        check('outer', df1.merge(df2, on='id', how='outer').shape == (3, 3))


class TestToNumericDatetimeTimedelta:
    def test_to_numeric(self):
        from fluidframe.core.fluidframe import to_numeric
        from fluidframe.core.ucs2column import UCS2Column_SoA
        col = UCS2Column_SoA(3); col.append('1'); col.append('2.5'); col.append('abc')
        r = to_numeric(col, errors='coerce')
        check('coerce', r[0] == 1 and r[1] == 2.5)

    def test_to_datetime(self):
        from fluidframe.core.fluidframe import to_datetime
        from fluidframe.core.ucs2column import UCS2Column_SoA
        col = UCS2Column_SoA(2); col.append('2023-01-15'); col.append('2023/06/20')
        r = to_datetime(col)
        check('to_datetime', r.length == 2)

    def test_to_date(self):
        from fluidframe.core.fluidframe import to_date
        from fluidframe.core.ucs2column import UCS2Column_SoA
        col = UCS2Column_SoA(2); col.append('2023-01-15'); col.append('20230620')
        r = to_date(col)
        check('to_date', r.length == 2)

    def test_to_time(self):
        from fluidframe.core.fluidframe import to_time
        from fluidframe.core.ucs2column import UCS2Column_SoA
        col = UCS2Column_SoA(2); col.append('10:30:45'); col.append('14:00')
        r = to_time(col)
        check('to_time', r.length == 2)


def run_all():
    print('=== FluidFrame Tests ===')
    for cls in [TestReadCsv, TestPickle, TestTsv, TestConcat, TestMerge,
                TestToNumericDatetimeTimedelta]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
