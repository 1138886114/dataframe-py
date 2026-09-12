import sys
sys.path.insert(0, '.')
from fluidframe.core.intcolumn import IntColumn
from fluidframe.core.categorycolumn import CategoryColumn
from fluidframe.core.fastdataframe import FastDataFrame

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


class TestCategoryColumnConstruction:
    def test_empty(self):
        cc = CategoryColumn(4)
        check('empty', len(cc) == 0)

    def test_append(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        check('length', len(cc) == 4)
        check('val 0', cc[0] == 'apple')
        check('val 1', cc[1] == 'banana')
        check('val 2', cc[2] == 'apple')

    def test_codes(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        codes = cc._cate.codes
        check('codes', [codes[i] for i in range(4)] == [0, 1, 0, 2])

    def test_categories(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        cats = cc._cate.categories
        check('categories', cats == ['apple', 'banana', 'cherry'])


class TestCategoryColumnOps:
    def test_concat(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        cc2 = CategoryColumn(4); cc2.append('banana'); cc2.append('date')
        cc.concat(cc2)
        check('concat len', len(cc) == 6)
        check('concat val', cc[4] == 'banana' and cc[5] == 'date')

    def test_copy(self):
        cc = CategoryColumn(4); cc.append('x'); cc.append('y')
        cc2 = cc.copy()
        check('copy len', len(cc2) == 2)
        check('copy val', cc2[0] == 'x' and cc2[1] == 'y')

    def test_eq_ne(self):
        cc1 = CategoryColumn(4); cc1.append('a'); cc1.append('b')
        cc2 = CategoryColumn(4); cc2.append('a'); cc2.append('b')
        cc3 = CategoryColumn(4); cc3.append('a'); cc3.append('c')
        check('eq', cc1 == cc2)
        check('ne', cc1 != cc3)
        check('ne non-column', cc1 != 'x')

    def test_isin(self):
        cc = CategoryColumn(4)
        cc.append('a'); cc.append('b'); cc.append('a'); cc.append('c')
        r = cc.isin({'a'})
        check('isin', [r[i] for i in range(4)] == [True, False, True, False])

    def test_where_mask(self):
        mask = IntColumn(4)
        mask.append(1); mask.append(0); mask.append(1); mask.append(0)
        cc = CategoryColumn(4)
        cc.append('x'); cc.append('x'); cc.append('x'); cc.append('x')
        r = cc.where_mask(mask, 'fill')
        check('where_mask', [r[i] for i in range(4)] == ['x', 'fill', 'x', 'fill'])

    def test_drop_row(self):
        cc = CategoryColumn(4)
        cc.append('a'); cc.append('b'); cc.append('c'); cc.append('d')
        cc.drop_row(1)
        check('drop_row len', len(cc) == 3)
        check('drop_row val', [cc[i] for i in range(3)] == ['a', 'c', 'd'])


class TestCategoryAccessor:
    def _make(self):
        g = CategoryColumn(10)
        for v in ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']:
            g.append(v)
        return g

    def test_codes_is_intcolumn(self):
        g = self._make()
        codes = g._cate.codes
        check('is IntColumn', isinstance(codes, IntColumn))
        check('codes values', list(codes) == [0, 1, 0, 2, 1, 0])

    def test_categories_copy(self):
        g = self._make()
        cats = g._cate.categories
        check('categories', cats == ['apple', 'banana', 'cherry'])
        cats.append('mutated')
        check('is copy', g._cate.categories == ['apple', 'banana', 'cherry'])

    def test_rename_categories_dict(self):
        g = self._make()
        g2 = g._cate.rename_categories({'apple': 'fruit_a', 'banana': 'fruit_b'})
        check('renamed cats', g2._cate.categories == ['fruit_a', 'fruit_b', 'cherry'])
        check('codes kept', list(g2._cate.codes) == list(g._cate.codes))
        check('original preserved', g._cate.categories == ['apple', 'banana', 'cherry'])

    def test_rename_categories_list(self):
        g = self._make()
        g2 = g._cate.rename_categories(['A', 'B', 'C'])
        check('renamed cats', g2._cate.categories == ['A', 'B', 'C'])
        check('values', [g2[i] for i in range(len(g2))] == ['A', 'B', 'A', 'C', 'B', 'A'])

    def test_rename_duplicate_guard(self):
        g = self._make()
        try:
            g._cate.rename_categories({'apple': 'banana'})
            check('duplicate raises', False)
        except ValueError:
            check('duplicate raises', True)

    def test_rename_inplace(self):
        g = CategoryColumn(4); g.append('apple'); g.append('banana')
        r = g._cate.rename_categories({'apple': 'x'}, inplace=True)
        check('returns None', r is None)
        check('mutated', g._cate.categories == ['x', 'banana'])

    def test_add_categories(self):
        g = self._make()
        g2 = g._cate.add_categories(['date', 'elderberry'])
        check('added', g2._cate.categories == ['apple', 'banana', 'cherry', 'date', 'elderberry'])
        g2.append('date')
        check('can append new', list(g2._cate.codes)[-1] == 3)
        check('original unchanged', 'date' not in g._cate.categories)

    def test_add_categories_dedup(self):
        g = self._make()
        g2 = g._cate.add_categories(['apple', 'date'])
        check('deduped', g2._cate.categories == ['apple', 'banana', 'cherry', 'date'])

    def test_set_categories(self):
        g = self._make()
        g2 = g._cate.set_categories(['apple', 'banana'])
        check('subset cats', g2._cate.categories == ['apple', 'banana', ''])
        check('removed mapped to empty', [g2[i] for i in range(len(g2))] == ['apple', 'banana', 'apple', '', 'banana', 'apple'])

    def test_set_categories_reorder(self):
        g = self._make()
        g2 = g._cate.set_categories(['cherry', 'apple', 'banana'])
        check('reordered', g2._cate.categories == ['cherry', 'apple', 'banana', ''])

    def test_set_categories_inplace(self):
        g = CategoryColumn(4)
        for v in ['apple', 'banana', 'apple', 'cherry']: g.append(v)
        r = g._cate.set_categories(['apple'], inplace=True)
        check('returns None', r is None)
        check('categories', g._cate.categories == ['apple', ''])

    def test_remove_categories(self):
        g = self._make()
        g2 = g._cate.remove_categories(['cherry'])
        check('removed', g2._cate.categories == ['apple', 'banana', ''])
        check('values', [g2[i] for i in range(len(g2))] == ['apple', 'banana', 'apple', '', 'banana', 'apple'])

    def test_remove_categories_inplace(self):
        g = CategoryColumn(4)
        for v in ['apple', 'banana', 'apple', 'cherry']: g.append(v)
        r = g._cate.remove_categories(['apple'], inplace=True)
        check('returns None', r is None)
        check('categories', g._cate.categories == ['banana', 'cherry', ''])
        check('values', [g[i] for i in range(len(g))] == ['', 'banana', '', 'cherry'])

    def test_empty_column(self):
        g = CategoryColumn(0)
        check('empty codes', len(g._cate.codes) == 0)
        check('empty categories', g._cate.categories == [])


class TestCategoryInDataFrame:
    def test_df_category(self):
        df = FastDataFrame({'C': 'category'})
        df.append_row(C='apple'); df.append_row(C='banana'); df.append_row(C='apple')
        check('val', df['C'][0] == 'apple' and df['C'][1] == 'banana')
        check('len', df.df_length == 3)

    def df_rename(self):
        df = FastDataFrame({'C': 'category'})
        df.append_row(C='apple'); df.append_row(C='banana'); df.append_row(C='apple')
        r = df['C']._cate.rename_categories({'apple': 'AA'})
        check('codes', list(r._cate.codes) == [0, 1, 0])
        check('val', r[0] == 'AA')


class TestCategoryColumnOwnership:
    def test_pop_front_raises_on_empty(self):
        cc = CategoryColumn(4)
        check('pop_front empty', expect_exception(IndexError, cc.pop_front))

    def test_pop_back_raises_on_empty(self):
        cc = CategoryColumn(4)
        check('pop_back empty', expect_exception(IndexError, cc.pop_back))

    def test_owning_pop_front_works(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        cc.pop_front()
        check('owning pop_front len', len(cc) == 3)
        check('owning pop_front val', cc[0] == 'banana')

    def test_owning_pop_back_works(self):
        cc = CategoryColumn(4)
        cc.append('apple'); cc.append('banana'); cc.append('apple'); cc.append('cherry')
        cc.pop_back()
        check('owning pop_back len', len(cc) == 3)
        check('owning pop_back val', cc[2] == 'apple')


def run_all():
    print('=== CategoryColumn Tests ===')
    for cls in [TestCategoryColumnConstruction, TestCategoryColumnOps, TestCategoryAccessor,
                TestCategoryInDataFrame, TestCategoryColumnOwnership]:
        for name in dir(cls):
            if name.startswith('test_'):
                getattr(cls(), name)()
    print(f'\nResults: {passed} passed, {failed} failed')
    if failed: sys.exit(1)

if __name__ == '__main__':
    run_all()
