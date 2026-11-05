#!/usr/bin/env python3
"""
ndframe 完整功能测试套件
构建: python3 -c "from setuptools import setup; from Cython.Build import cythonize; setup(ext_modules=cythonize('ndframe.pyx', language_level=3))" build_ext --build-lib=build
用法: python ndframe_test.py
"""
import sys, os, json, re, math, random, time
from typing import List, Tuple, Optional
from collections import Counter

test_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(test_dir, "build")
if os.path.isdir(build_dir):
    sys.path.insert(0, build_dir)

try:
    from ndframe import NDframe
except ImportError:
    print("找不到 ndframe 模块，请先编译:")
    print("  cd ndframe && python setup.py build_ext --build-lib=build")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
# 测试基础设施
# ════════════════════════════════════════════════════════════════
PASS = FAIL = 0

def check(desc: str, actual, expected):
    global PASS, FAIL
    if actual == expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {desc} — expected {expected!r}, got {actual!r}")

def section(name: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

def subsection(name: str):
    print(f"\n--- {name}")

# ════════════════════════════════════════════════════════════════
# 测试数据加载
# ════════════════════════════════════════════════════════════════
TEST_JSON_RAW = '{"description": "识别成功", "error_code": 0, "recognize_time": 882, "request_id": "d99b058416de11f180436c92bf76ee27", "result": [{"position": {"height": 24, "left": 152, "top": 31, "width": 20}, "quad": "152,31,172,31,172,55,152,55", "score": 0.59803832, "type": "print", "words": "物"}, {"position": {"height": 41, "left": 180, "top": 16, "width": 85}, "quad": "180,16,265,16,265,57,180,57", "score": 0.97882873, "type": "handwritting", "words": "3547"}, {"position": {"height": 49, "left": 268, "top": 10, "width": 81}, "quad": "268,15,346,10,349,53,271,59", "score": 0.9176501, "type": "handwritting", "words": "(SW)"}, {"position": {"height": 49, "left": 562, "top": 35, "width": 199}, "quad": "562,35,761,35,761,84,562,84", "score": 0.70832467, "type": "print", "words": "镜湖馨院"}, {"position": {"height": 47, "left": 777, "top": 35, "width": 42}, "quad": "779,35,819,37,817,82,777,80", "score": 0.99970657, "type": "print", "words": "收"}, {"position": {"height": 41, "left": 842, "top": 37, "width": 38}, "quad": "842,37,880,37,880,78,842,78", "score": 0.9473635, "type": "print", "words": "据"}, {"position": {"height": 22, "left": 1030, "top": 23, "width": 137}, "quad": "1030,23,1167,23,1167,45,1030,45", "score": 0.99425405, "type": "print", "words": "澳门镜湖马路"}, {"position": {"height": 20, "left": 1194, "top": 27, "width": 208}, "quad": "1194,27,1402,27,1402,47,1194,47", "score": 0.99275136, "type": "print", "words": "Estrada do Repouso, Macau"}, {"position": {"height": 18, "left": 1028, "top": 68, "width": 139}, "quad": "1028,68,1167,68,1167,86,1028,86", "score": 0.98288333, "type": "print", "words": "Tel:(853)283 71333"}, {"position": {"height": 20, "left": 1194, "top": 68, "width": 202}, "quad": "1194,68,1396,68,1396,88,1194,88", "score": 0.99970645, "type": "print", "words": "http://www.kwh.org.mo"}, {"position": {"height": 29, "left": 101, "top": 92, "width": 105}, "quad": "101,92,206,92,206,121,101,121", "score": 0.95950699, "type": "print", "words": "第一门诊"}, {"position": {"height": 66, "left": 252, "top": 75, "width": 165}, "quad": "252,75,417,75,417,141,252,141", "score": 0.64164132, "type": "handwritting", "words": "6-25-976"}, {"position": {"height": 25, "left": 562, "top": 90, "width": 201}, "quad": "562,90,763,90,763,115,562,115", "score": 0.96601856, "type": "print", "words": "KIANG WU HOSPITAL"}, {"position": {"height": 29, "left": 779, "top": 88, "width": 105}, "quad": "779,88,884,88,884,117,779,117", "score": 0.98437411, "type": "print", "words": "Receipt"}, {"position": {"height": 24, "left": 1028, "top": 109, "width": 99}, "quad": "1028,109,1127,109,1127,133,1028,133", "score": 0.99908417, "type": "print", "words": "收据号吗"}, {"position": {"height": 20, "left": 1137, "top": 113, "width": 83}, "quad": "1137,113,1220,113,1220,133,1137,133", "score": 0.97683352, "type": "print", "words": "Receipt No."}, {"position": {"height": 18, "left": 1230, "top": 109, "width": 136}, "quad": "1230,109,1366,109,1366,127,1230,127", "score": 0.98516875, "type": "print", "words": "000021287824"}, {"position": {"height": 24, "left": 107, "top": 156, "width": 47}, "quad": "107,156,154,156,154,180,107,180", "score": 0.9746902, "type": "print", "words": "姓名"}, {"position": {"height": 41, "left": 266, "top": 139, "width": 117}, "quad": "266,139,383,139,383,180,266,180", "score": 0.78784996, "type": "handwritting", "words": "梁丽芬"}, {"position": {"height": 45, "left": 496, "top": 140, "width": 215}, "quad": "496,140,711,140,711,185,496,185", "score": 0.54934788, "type": "handwritting", "words": "D91:18May2025"}, {"position": {"height": 22, "left": 1078, "top": 158, "width": 51}, "quad": "1078,158,1129,158,1129,180,1078,180", "score": 0.99856591, "type": "print", "words": "科别"}, {"position": {"height": 32, "left": 1133, "top": 148, "width": 65}, "quad": "1133,148,1198,148,1198,180,1133,180", "score": 0.53711998, "type": "handwritting", "words": "外科"}, {"position": {"height": 17, "left": 107, "top": 184, "width": 49}, "quad": "107,184,156,184,156,201,107,201", "score": 0.84206271, "type": "print", "words": "Name"}, {"position": {"height": 20, "left": 263, "top": 191, "width": 168}, "quad": "263,191,431,191,431,211,263,211", "score": 0.96243566, "type": "print", "words": "LEONG LAI FAN"}, {"position": {"height": 21, "left": 1080, "top": 184, "width": 47}, "quad": "1080,184,1127,184,1127,205,1080,205", "score": 0.9935149, "type": "print", "words": "Dept."}, {"position": {"height": 22, "left": 107, "top": 232, "width": 97}, "quad": "107,232,204,232,204,254,107,254", "score": 0.99695909, "type": "print", "words": "病历号吗"}, {"position": {"height": 25, "left": 259, "top": 225, "width": 101}, "quad": "259,225,360,225,360,250,259,250", "score": 0.6683476, "type": "handwritting", "words": "422870"}, {"position": {"height": 24, "left": 1028, "top": 232, "width": 99}, "quad": "1028,232,1127,232,1127,256,1028,256", "score": 0.99852163, "type": "print", "words": "诊治日期"}, {"position": {"height": 27, "left": 1139, "top": 223, "width": 136}, "quad": "1139,223,1275,223,1275,250,1139,250", "score": 0.31400001, "type": "handwritting", "words": "20/5/2025"}, {"position": {"height": 19, "left": 107, "top": 258, "width": 138}, "quad": "107,258,245,258,245,277,107,277", "score": 0.97855914, "type": "print", "words": "Medical Record No."}, {"position": {"height": 17, "left": 836, "top": 260, "width": 105}, "quad": "836,260,941,260,941,277,836,277", "score": 0.54761314, "type": "handwritting", "words": "W250013046"}, {"position": {"height": 17, "left": 1028, "top": 260, "width": 99}, "quad": "1028,260,1127,260,1127,277,1028,277", "score": 0.58576506, "type": "handwritting", "words": "Date incured"}, {"position": {"height": 23, "left": 109, "top": 313, "width": 47}, "quad": "109,313,156,313,156,336,109,336", "score": 0.99905711, "type": "print", "words": "代号"}, {"position": {"height": 25, "left": 358, "top": 313, "width": 97}, "quad": "358,313,455,313,455,338,358,338", "score": 0.98840278, "type": "print", "words": "费用摘要"}, {"position": {"height": 25, "left": 651, "top": 313, "width": 49}, "quad": "651,313,700,313,700,338,651,338", "score": 0.9856863, "type": "print", "words": "金额"}, {"position": {"height": 25, "left": 773, "top": 313, "width": 48}, "quad": "773,313,821,313,821,338,773,338", "score": 0.9989959, "type": "print", "words": "代号"}, {"position": {"height": 25, "left": 1028, "top": 313, "width": 99}, "quad": "1028,313,1127,313,1127,338,1028,338", "score": 0.99648768, "type": "print", "words": "费用摘要"}, {"position": {"height": 24, "left": 1317, "top": 316, "width": 49}, "quad": "1317,316,1366,316,1366,340,1317,340", "score": 0.99178386, "type": "print", "words": "金额"}, {"position": {"height": 19, "left": 109, "top": 340, "width": 47}, "quad": "109,340,156,340,156,359,109,359", "score": 0.99550813, "type": "print", "words": "Code"}, {"position": {"height": 19, "left": 358, "top": 340, "width": 95}, "quad": "358,340,453,340,453,359,358,359", "score": 0.98221356, "type": "print", "words": "Particulars"}, {"position": {"height": 17, "left": 651, "top": 342, "width": 59}, "quad": "651,342,710,342,710,359,651,359", "score": 0.96483034, "type": "print", "words": "Amount"}, {"position": {"height": 17, "left": 773, "top": 342, "width": 46}, "quad": "773,342,819,342,819,359,773,359", "score": 0.6813159, "type": "handwritting", "words": "Code"}, {"position": {"height": 19, "left": 1028, "top": 342, "width": 95}, "quad": "1028,342,1123,342,1123,361,1028,361", "score": 0.9710052, "type": "print", "words": "Particulars"}, {"position": {"height": 18, "left": 1317, "top": 342, "width": 59}, "quad": "1317,342,1376,342,1376,360,1317,360", "score": 0.98494464, "type": "print", "words": "Amount"}, {"position": {"height": 22, "left": 123, "top": 363, "width": 29}, "quad": "123,363,152,363,152,385,123,385", "score": 0.98995531, "type": "print", "words": "11"}, {"position": {"height": 28, "left": 340, "top": 361, "width": 30}, "quad": "340,361,370,361,370,389,340,389", "score": 0.99955475, "type": "print", "words": "诊"}, {"position": {"height": 26, "left": 423, "top": 363, "width": 30}, "quad": "423,363,453,363,453,389,423,389", "score": 0.99982858, "type": "print", "words": "金"}, {"position": {"height": 24, "left": 631, "top": 363, "width": 69}, "quad": "631,363,700,363,700,387,631,387", "score": 0.50246966, "type": "handwritting", "words": "60.00"}, {"position": {"height": 24, "left": 787, "top": 363, "width": 32}, "quad": "787,363,819,363,819,387,787,387", "score": 0.89684486, "type": "print", "words": "20"}, {"position": {"height": 28, "left": 1001, "top": 361, "width": 31}, "quad": "1001,361,1032,361,1032,389,1001,389", "score": 0.99969578, "type": "print", "words": "药"}, {"position": {"height": 26, "left": 1094, "top": 363, "width": 31}, "quad": "1094,363,1125,363,1125,389,1094,389", "score": 0.99978113, "type": "print", "words": "费"}, {"position": {"height": 22, "left": 1295, "top": 363, "width": 71}, "quad": "1295,363,1366,363,1366,385,1295,385", "score": 0.79398894, "type": "handwritting", "words": "94.00"}, {"position": {"height": 25, "left": 121, "top": 397, "width": 31}, "quad": "121,397,152,397,152,422,121,422", "score": 0.97897315, "type": "print", "words": "31"}, {"position": {"height": 31, "left": 340, "top": 396, "width": 115}, "quad": "340,396,455,396,455,427,340,427", "score": 0.98985696, "type": "print", "words": "治 疗费"}, {"position": {"height": 25, "left": 619, "top": 397, "width": 89}, "quad": "619,397,708,397,708,422,619,422", "score": 0.64955848, "type": "handwritting", "words": "400.00"}, {"position": {"height": 25, "left": 787, "top": 397, "width": 32}, "quad": "787,397,819,397,819,422,787,422", "score": 0.9076125, "type": "print", "words": "33"}, {"position": {"height": 29, "left": 1005, "top": 395, "width": 120}, "quad": "1005,395,1125,395,1125,424,1005,424", "score": 0.9922967, "type": "print", "words": "材 料 费"}, {"position": {"height": 25, "left": 1287, "top": 397, "width": 87}, "quad": "1287,397,1374,397,1374,422,1287,422", "score": 0.47724807, "type": "handwritting", "words": "305.00"}, {"position": {"height": 24, "left": 107, "top": 580, "width": 99}, "quad": "107,580,206,580,206,604,107,604", "score": 0.99945474, "type": "print", "words": "合计金额"}, {"position": {"height": 20, "left": 214, "top": 584, "width": 102}, "quad": "214,584,316,584,316,604,214,604", "score": 0.39175224, "type": "handwritting", "words": "Total Amount"}, {"position": {"height": 24, "left": 342, "top": 574, "width": 172}, "quad": "342,574,514,574,514,598,342,598", "score": 0.82613277, "type": "print", "words": "MOP 859.00"}, {"position": {"height": 16, "left": 749, "top": 588, "width": 10}, "quad": "749,588,759,588,759,604,749,604", "score": 0.93477511, "type": "print", "words": "*"}, {"position": {"height": 23, "left": 761, "top": 586, "width": 101}, "quad": "761,586,862,586,862,609,761,609", "score": 0.67980844, "type": "handwritting", "words": "处方日起话"}, {"position": {"height": 23, "left": 866, "top": 586, "width": 105}, "quad": "866,586,971,586,971,609,866,609", "score": 0.75366873, "type": "handwritting", "words": "[乐物逾期7"}, {"position": {"height": 23, "left": 1018, "top": 586, "width": 240}, "quad": "1018,586,1258,586,1258,609,1018,609", "score": 0.64285058, "type": "handwritting", "words": "取，一律作废不作退款。"}, {"position": {"height": 14, "left": 1173, "top": 693, "width": 15}, "quad": "1173,693,1188,693,1188,707,1173,707", "score": 0.53198606, "type": "print", "words": "2"}, {"position": {"height": 32, "left": 1190, "top": 692, "width": 50}, "quad": "1190,692,1240,692,1240,724,1190,724", "score": 0.99476331, "type": "print", "words": "HN"}, {"position": {"height": 34, "left": 1246, "top": 701, "width": 47}, "quad": "1246,701,1293,701,1293,735,1246,735", "score": 0.40249404, "type": "handwritting", "words": "2025"}, {"position": {"height": 21, "left": 95, "top": 799, "width": 405}, "quad": "95,799,500,799,500,820,95,820", "score": 0.97881371, "type": "print", "words": "1.离开前请核对收据上资料，如有错漏，从速指正。"}, {"position": {"height": 27, "left": 605, "top": 807, "width": 152}, "quad": "605,807,757,807,757,834,605,834", "score": 0.99337685, "type": "print", "words": "领药号：1204"}, {"position": {"height": 16, "left": 1030, "top": 830, "width": 67}, "quad": "1030,830,1097,830,1097,846,1030,846", "score": 0.98842448, "type": "print", "words": "Cashier"}, {"position": {"height": 18, "left": 117, "top": 824, "width": 367}, "quad": "117,824,484,824,484,842,117,842", "score": 0.9139486, "type": "print", "words": "Please confirm before you leave the counter."}, {"position": {"height": 25, "left": 1030, "top": 801, "width": 69}, "quad": "1030,801,1099,801,1099,826,1030,826", "score": 0.7139855, "type": "handwritting", "words": "收费员"}, {"position": {"height": 24, "left": 1119, "top": 791, "width": 83}, "quad": "1119,791,1202,791,1202,815,1119,815", "score": 0.51912945, "type": "handwritting", "words": "84329"}, {"position": {"height": 21, "left": 95, "top": 846, "width": 265}, "quad": "95,846,360,846,360,867,95,867", "score": 0.98743612, "type": "print", "words": "2.此收据经涂改无效，遗失不补。"}, {"position": {"height": 24, "left": 1030, "top": 869, "width": 93}, "quad": "1030,869,1123,869,1123,893,1030,893", "score": 0.99321455, "type": "print", "words": "收费时间"}, {"position": {"height": 27, "left": 1147, "top": 856, "width": 241}, "quad": "1147,856,1388,856,1388,883,1147,883", "score": 0.86476594, "type": "handwritting", "words": "29/5/2025 9:25:58"}, {"position": {"height": 20, "left": 117, "top": 871, "width": 601}, "quad": "117,871,718,871,718,891,117,891", "score": 0.93447637, "type": "print", "words": "Official receipt is invalid when being altered and it wil not be issued again."}, {"position": {"height": 21, "left": 95, "top": 895, "width": 809}, "quad": "95,895,904,895,904,916,95,916", "score": 0.95458394, "type": "print", "words": "3.因故未能进行检查或治疗，可于激费日期起两个月内凭收据办理退款(需扣除有关手续费)，逾期不予接受。"}, {"position": {"height": 21, "left": 1030, "top": 895, "width": 97}, "quad": "1030,895,1127,895,1127,916,1030,916", "score": 0.69826406, "type": "handwritting", "words": "Time of payfe"}, {"position": {"height": 18, "left": 117, "top": 920, "width": 789}, "quad": "117,920,906,920,906,938,117,938", "score": 0.92379206, "type": "print", "words": "In case any medical examinations or treatments were cancelled, a refund with a certain amount"}, {"position": {"height": 19, "left": 117, "top": 942, "width": 789}, "quad": "117,942,906,942,906,961,117,961", "score": 0.94473153, "type": "print", "words": "of handing fee deducted may be given on presentation of the receipt of payment if the refund"}, {"position": {"height": 24, "left": 989, "top": 955, "width": 118}, "quad": "989,955,1107,955,1107,979,989,979", "score": 0.99779457, "type": "print", "words": "(扫码认证)"}, {"position": {"height": 18, "left": 117, "top": 963, "width": 500}, "quad": "117,963,617,963,617,981,117,981", "score": 0.93332303, "type": "print", "words": "request is made within fwo months from the date of payment."}, {"position": {"height": 21, "left": 93, "top": 985, "width": 198}, "quad": "93,985,291,985,291,1006,93,1006", "score": 0.9737187, "type": "print", "words": "4.院方保留最终解释权。"}, {"position": {"height": 21, "left": 117, "top": 1010, "width": 579}, "quad": "117,1010,696,1010,696,1031,117,1031", "score": 0.93145478, "type": "print", "words": "Kiang Wu Hospital reserves the right to amend the terms and conditions."}]}'

# row_id → dict of parsed fields (pre-computed for expected-value assertions)
ParsedRow = dict

def _parse_one(i: int, r: dict) -> ParsedRow:
    pos = r["position"]
    left = pos["left"]
    top = pos["top"]
    w = pos["width"]
    h = pos["height"]
    return {
        "idx": i,
        "x0": left,
        "y0": top,
        "x1": left + w,
        "y1": top + h,
        "width": w,
        "height": h,
        "page": 1,
        "flags": 0,
        "text": r.get("words", ""),
        "text_sep": "",
    }


def load_test_data() -> Tuple[NDframe, List[ParsedRow]]:
    """解析 test_text JSON 并构建 NDframe，返回 (df, rows)"""
    data = json.loads(TEST_JSON_RAW)
    results = data["result"]
    n = len(results)
    rows = [_parse_one(i, r) for i, r in enumerate(results)]
    df = NDframe(n)
    for i, r in enumerate(results):
        pos = r["position"]
        left = pos["left"]
        top = pos["top"]
        w = pos["width"]
        h = pos["height"]
        df.append(i, left, top, left + w, top + h, w, h, 1, 0, r.get("words", ""), "")
    return df, rows


def make_small_df() -> NDframe:
    """构造小型标准测试 DF（用于需要精确控制特征值的测试）"""
    df = NDframe()
    df.append(0, 10, 20, 100, 60, 90, 40, 1, 0, "hello world 你好世界", "sep_a")
    df.append(1, 50, 80, 200, 150, 150, 70, 2, 1, "foo bar baz ", "sep_b_connector")
    df.append(2, 30, 40, 80, 90, 50, 50, 1, 0, "test 123 regex 正则测试", "sep_c")
    df.append(3, 5, 5, 30, 30, 25, 25, 3, 2, "short 快 短", "sep_d")
    df.append(4, 60, 100, 180, 200, 120, 100, 1, 0, "another one 这一个", "sep_e")
    return df


# ════════════════════════════════════════════════════════════════
# 测试套件
# ════════════════════════════════════════════════════════════════

def test_basic_properties(df: NDframe, rows: List[ParsedRow]):
    section("1. 基础构造与属性")
    check("len(df)", len(df), len(rows))
    check("df.empty", df.empty, False)
    check("bool(df)", bool(df), True)

    empty_df = NDframe()
    check("empty df.empty", empty_df.empty, True)
    check("bool(empty)", bool(empty_df), False)

    check("df[0]['x0']", df[0]["x0"], rows[0]["x0"])
    check("df[0]['y0']", df[0]["y0"], rows[0]["y0"])
    check("df[0]['x1']", df[0]["x1"], rows[0]["x1"])
    check("df[0]['y1']", df[0]["y1"], rows[0]["y1"])
    check("df[0]['text']", df[0]["text"], rows[0]["text"])

    df_first = df[0]
    r0 = rows[0]
    check("df[0] dict keys", set(df_first.keys()), {"idx", "x0", "y0", "x1", "y1", "width", "height", "page", "flags", "text", "text_sep"})

    # Negative index
    check("df[-1] idx", df[-1]["idx"], rows[-1]["idx"])
    check("df[-2] text", df[-2]["text"], rows[-2]["text"])

    # Slice
    sliced = df[1:5]
    check("df[1:5] len", len(sliced), 4)
    check("df[1:5][0] text", sliced[0]["text"], rows[1]["text"])

    nd = df[3:3]
    check("df[3:3] empty", len(nd), 0)


def test_to_list_string(df: NDframe, rows: List[ParsedRow]):
    section("2. to_list / to_string / print")
    lst = df.to_list()
    check("to_list len", len(lst), len(rows))
    check("to_list[0]['text']", lst[0]["text"], rows[0]["text"])
    s = df.to_string()
    check("to_string produces output", len(s) > 100, True)
    check("to_string contains row count", str(len(rows)) in s, True)

    rep = repr(df)
    check("__repr__", rep, f"NDframe({len(rows)} rows)")
    check("__str__", str(df), f"<NDframe: {len(rows)} rows>")

    # data property
    check("df.data len", len(df.data), len(rows))
    check("df.data[0]['text']", df.data[0]["text"], rows[0]["text"])

    # iter
    it = list(df)
    check("list(df) len", len(it), len(rows))
    check("list(df)[0]['text']", it[0]["text"], rows[0]["text"])


# ──────────────────────────────────────────────────────────────
# NumericAccessor 测试组
# ──────────────────────────────────────────────────────────────

def test_numeric_aggregation(df: NDframe, rows: List[ParsedRow]):
    section("3. NumericAccessor 聚合")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]
    x1_vals = [r["x1"] for r in rows]
    w_vals = [r["width"] for r in rows]
    h_vals = [r["height"] for r in rows]

    check("x0.max", df.x0.max(), max(x0_vals))
    check("x0.min", df.x0.min(), min(x0_vals))
    check("y0.max", df.y0.max(), max(y0_vals))
    check("y0.min", df.y0.min(), min(y0_vals))
    check("x1.max", df.x1.max(), max(x1_vals))
    check("x1.min", df.x1.min(), min(x1_vals))
    check("width.max", df.width.max(), max(w_vals))
    check("width.min", df.width.min(), min(w_vals))
    check("height.max", df.height.max(), max(h_vals))
    check("height.min", df.height.min(), min(h_vals))

    check("x0.mean", round(df.x0.mean(), 6), round(sum(x0_vals) / len(x0_vals), 6))
    check("y0.mean", round(df.y0.mean(), 6), round(sum(y0_vals) / len(y0_vals), 6))
    check("height.mean", round(df.height.mean(), 6), round(sum(h_vals) / len(h_vals), 6))

    # field_name
    check("x0.field_name", df.x0.field_name, "x0")
    check("y0.field_name", df.y0.field_name, "y0")
    check("height.field_name", df.height.field_name, "height")
    check("repr(x0)", repr(df.x0).startswith("<Numeric x0:"), True)


def test_numeric_filter(df: NDframe, rows: List[ParsedRow]):
    section("4. NumericAccessor 过滤")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]
    w_vals = [r["width"] for r in rows]
    h_vals = [r["height"] for r in rows]

    # greater
    gt40 = sum(1 for v in x0_vals if v > 40)
    check("x0.greater(40)", len(df.x0.greater(40)), gt40)

    lt40 = sum(1 for v in x0_vals if v < 40)
    check("x0.less(40)", len(df.x0.less(40)), lt40)

    # between
    mid_y0 = sum(1 for v in y0_vals if 50 < v < 800)
    check("y0.between(50,800)", len(df.y0.between(50, 800)), mid_y0)

    # values
    check("x0.values == x0 list", list(df.x0.values()), x0_vals)
    check("x0.values(0)", df.x0.values(0), x0_vals[0])
    check("x0.values(-1)", df.x0.values(-1), x0_vals[-1])


def test_numeric_arithmetic(df: NDframe, rows: List[ParsedRow]):
    section("5. NumericAccessor 运算")
    x0_vals = [r["x0"] for r in rows]

    df_a = df.copy()
    df_a.x0.add(5)
    check("x0.add(5)", list(df_a.x0.values()), [v + 5 for v in x0_vals])

    df_s = df.copy()
    df_s.x0.sub(3)
    check("x0.sub(3)", list(df_s.x0.values()), [max(v - 3, 0) for v in x0_vals])

    # diff
    raw_diff = [None] + [x0_vals[i] - x0_vals[i-1] for i in range(1, len(x0_vals))]
    check("x0.diff", df.x0.diff(), raw_diff)

    # var / std
    mean_x = sum(x0_vals) / len(x0_vals)
    var1 = sum((v - mean_x) ** 2 for v in x0_vals) / (len(x0_vals) - 1)
    var0 = sum((v - mean_x) ** 2 for v in x0_vals) / len(x0_vals)
    check("x0.var(ddof=1)", round(df.x0.var(1), 10), round(var1, 10))
    check("x0.var(ddof=0)", round(df.x0.var(0), 10), round(var0, 10))
    check("x0.std(ddof=1)", round(df.x0.std(1), 10), round(math.sqrt(var1), 10))

    # mode
    from collections import Counter
    c = Counter(x0_vals)
    mode_val, mode_cnt = min((v for v in c.items()), key=lambda x: (-x[1], x[0]))
    check("x0.mode", df.x0.mode(), (mode_val, mode_cnt))

    # empty df stats
    empty_df = NDframe()
    check("empty var", empty_df.x0.var(), 0.0)
    check("empty std", empty_df.x0.std(), 0.0)
    check("empty mean", empty_df.x0.mean(), 0.0)
    check("empty diff", empty_df.x0.diff(), [])


def test_numeric_sort(df: NDframe, rows: List[ParsedRow]):
    section("6. NumericAccessor 排序")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]
    width_vals = [r["width"] for r in rows]

    df_a = df.copy()
    df_a.x0.sort()
    check("x0.sort asc", list(df_a.x0.values()), sorted(x0_vals))

    df_b = df.copy()
    df_b.y0.sort(descending=True)
    check("y0.sort desc", list(df_b.y0.values()), sorted(y0_vals, reverse=True))

    df_c = df.copy()
    df_c.width.sort()
    check("width.sort asc", list(df_c.width.values()), sorted(width_vals))

    # sort_by convenience
    df_d = df.copy()
    df_d.sort_by_x0(descending=True)
    check("sort_by_x0 desc", list(df_d.x0.values()), sorted(x0_vals, reverse=True))

    df_e = df.copy()
    df_e.sort_by_y0()
    check("sort_by_y0 asc", list(df_e.y0.values()), sorted(y0_vals))

    df_f = df.copy()
    df_f.sort_by("x1")
    x1_vals = [r["x1"] for r in rows]
    check("sort_by('x1')", list(df_f.x1.values()), sorted(x1_vals))

    df_g = df.copy()
    df_g.sort_by("height", descending=True)
    h_vals = [r["height"] for r in rows]
    check("sort_by('height' desc)", list(df_g.height.values()), sorted(h_vals, reverse=True))

    # chained sort
    df_h = df.copy()
    df_h.x0.sort().y0.sort()
    check("chained sort len", len(df_h), len(rows))


def test_numeric_idx(df: NDframe, rows: List[ParsedRow]):
    section("7. NumericAccessor idx_* 索引")
    x0_vals = [r["x0"] for r in rows]
    y0_vals = [r["y0"] for r in rows]

    expected_gt = [i for i, v in enumerate(x0_vals) if v > 40]
    expected_lt = [i for i, v in enumerate(x0_vals) if v < 40]
    expected_eq = [i for i, v in enumerate(x0_vals) if v == 30]

    check("idx_greater(40)", df.x0.idx_greater(40), expected_gt)
    check("idx_less(40)", df.x0.idx_less(40), expected_lt)
    check("idx_equal(30)", df.x0.idx_equal(30), expected_eq)

    # between
    expected_btw = [i for i, v in enumerate(y0_vals) if 50 < v < 800]
    check("y0.idx_between(50,800)", df.y0.idx_between(50, 800), expected_btw)


# ──────────────────────────────────────────────────────────────
# StringAccessor 测试组
# ──────────────────────────────────────────────────────────────

def test_string_basic(df: NDframe, rows: List[ParsedRow]):
    section("8. StringAccessor 基础")
    texts = [r["text"] for r in rows]

    check("text.field_name", df.text.field_name, "text")
    check("text_sep.field_name", df.text_sep.field_name, "text_sep")
    check("text.join", df.text.join("|"), "|".join(texts))
    check("text.len", df.text.len(), [len(t) for t in texts])
    check("text.values", df.text.values(), texts)
    check("text.values(0)", df.text.values(0), texts[0])
    check("text.values(-1)", df.text.values(-1), texts[-1])

    check("list(text) == text.values()", list(df.text), df.text.values())
    print("df.text.is_chinese()=",df.text.is_chinese().print())
    print("df.text.is_numeric_punct()=",df.text.is_numeric_punct().print())

    check("repr(text) starts with 'text['", repr(df.text).startswith("text["), True)


def test_string_transform(df: NDframe, rows: List[ParsedRow]):
    section("9. StringAccessor 变换")
    texts = [r["text"] for r in rows]

    # replace
    df_r = df.copy()
    df_r.text.replace(" ", "_")
    check("text.replace(' ', '_')",
          df_r.text.join("|"),
          "|".join(t.replace(" ", "_") for t in texts))

    # strip / rstrip / lstrip
    df_st = df.copy()
    df_st.text.strip()
    check("text.strip endswith space", [df_st.text.values(i)[-1] != " " for i in range(len(rows)) if texts[i].endswith(" ")], [True] * sum(1 for t in texts if t.endswith(" ")))

    # capitalize
    df_cap = df.copy()
    df_cap.text.capitalize()
    check("text.capitalize[0] first char", df_cap.text.values(0)[0], texts[0][0].upper() if texts[0] else "")
    check("text.capitalize[0] rest lowercase", df_cap.text.values(0)[1:5], texts[0][1:5].lower() if len(texts[0]) > 1 else texts[0][1:5])

    # swapcase
    df_swp = df.copy()
    df_swp.text.swapcase()
    check("text.swapcase", df_swp.text.values(0), texts[0].swapcase())

    # slice
    sliced = df.text.slice(0, 5)
    check("text.slice(0,5)[0]", sliced[0], texts[0][0:5])

    # count
    total_e = sum(t.count("e") for t in texts)
    check("text.count('e')", df.text.count("e"), total_e)
    check("text.count('xyz')", df.text.count("xyz"), 0)


def test_string_is(df: NDframe, rows: List[ParsedRow]):
    section("10. StringAccessor 字符类型判断")
    texts = [r["text"] for r in rows]

    check("text.isalnum", len(df.text.isalnum()), sum(1 for t in texts if t and t.isalnum()))
    check("text.isalpha", len(df.text.isalpha()), sum(1 for t in texts if t and t.isalpha()))
    check("text.isdigit", len(df.text.isdigit()), sum(1 for t in texts if t and t.isdigit()))
    check("text.isspace", len(df.text.isspace()), sum(1 for t in texts if t and t.isspace()))
    check("text.islower", len(df.text.islower()), sum(1 for t in texts if t and t.islower()))
    check("text.isupper", len(df.text.isupper()), sum(1 for t in texts if t and t.isupper()))
    check("text.isnumeric", len(df.text.isnumeric()), sum(1 for t in texts if t and t.isnumeric()))
    check("text.isdecimal", len(df.text.isdecimal()), sum(1 for t in texts if t and t.isdecimal()))


def test_string_starts_ends(df: NDframe, rows: List[ParsedRow]):
    section("11. StringAccessor startswith / endswith")
    texts = [r["text"] for r in rows]

    check("text.startswith('收')", len(df.text.startswith("收")), sum(1 for t in texts if t.startswith("收")))
    check("text.startswith('MOP')", len(df.text.startswith("MOP")), sum(1 for t in texts if t.startswith("MOP")))
    check("text.startswith('nonexist')", len(df.text.startswith("nonexist")), 0)
    check("text.endswith('.')", len(df.text.endswith(".")), sum(1 for t in texts if t.endswith(".")))
    check("text.endswith('。')", len(df.text.endswith("。")), sum(1 for t in texts if t.endswith("。")))
    # endswith('') — note: empty string as suffix: Python str.endswith('') returns
    # True for any string, including "". But the implementation checks `if txt` first,
    # so empty text_sep rows are excluded.
    non_empty_sep = sum(1 for r in rows if r["text_sep"])
    check("text_sep.endswith('') non-empty rows", len(df.text_sep.endswith("")), non_empty_sep)


def test_string_regex(df: NDframe, rows: List[ParsedRow]):
    section("12. StringAccessor 正则")
    texts = [r["text"] for r in rows]

    r_数字 = re.compile(r"\d+")
    check("contains(数字)", len(df.text.contains(r_数字)), sum(1 for t in texts if r_数字.search(t)))

    r_收据 = re.compile(r"收据")
    check("contains(收据)", len(df.text.contains(r_收据)), sum(1 for t in texts if r_收据.search(t)))

    # regex_replace
    df_rr = df.copy()
    df_rr.text.regex_replace(re.compile(r"\d+"), "X")
    check("regex_replace 数字→X[0]", "X" in df_rr.text.values(0), bool(r_数字.search(texts[0])))

    # regex_extract — 返回仅含匹配行的新 NDframe，text 替换为第一个捕获组
    r_extract = re.compile(r"(\d+)")
    df_re = df.copy()
    res = df_re.text.regex_extract(r_extract)
    expected_count = sum(1 for t in texts if r_extract.search(t))
    check("regex_extract len", len(res), expected_count)
    expected_first = r_extract.search(next(t for t in texts if r_extract.search(t))).group(1)
    check("regex_extract first value", res.text.values(0), expected_first)


def test_string_len_filter(df: NDframe, rows: List[ParsedRow]):
    section("13. StringAccessor 长度过滤")
    texts = [r["text"] for r in rows]
    lengths = [len(t) for t in texts]

    check("strlen_greater(10)", len(df.text.strlen_greater(10)), sum(1 for l in lengths if l > 10))
    check("strlen_less(10)", len(df.text.strlen_less(10)), sum(1 for l in lengths if l < 10))
    check("strlen_equal(15)", len(df.text.strlen_equal(15)), sum(1 for l in lengths if l == 15))

    # regex + length combo
    r_all = re.compile(r".")
    check("regex_strlen_greater(10)", len(df.text.regex_strlen_greater(r_all, 10)), sum(1 for l in lengths if l > 10))
    check("regex_strlen_less(10)", len(df.text.regex_strlen_less(r_all, 10)), sum(1 for l in lengths if l < 10))
    check("regex_strlen_equal(15)", len(df.text.regex_strlen_equal(r_all, 15)), sum(1 for l in lengths if l == 15))

    r_num = re.compile(r"\d+")
    expected = sum(1 for i, t in enumerate(texts) if len(t) > 10 and r_num.search(t))
    check("regex_strlen_greater(10) with 数字", len(df.text.regex_strlen_greater(r_num, 10)), expected)


def test_string_set_ops(df: NDframe, rows: List[ParsedRow]):
    section("14. StringAccessor 集合筛选")
    texts = [r["text"] for r in rows]
    target_set = ["收", "据", "姓名", "nonexistent"]
    check("is_in", len(df.text.is_in(target_set)), sum(1 for t in texts if t in target_set))
    check("not_in", len(df.text.not_in(["收", "据"])), sum(1 for t in texts if t not in ["收", "据"]))

    check("contains_any True", df.text.contains_any(["收据", "nope"]), any(any(sub in t for sub in ["收据", "nope"]) for t in texts))
    check("contains_any False", df.text.contains_any(["ZZZZNONEXIST"]), False)


def test_string_idx_ops(df: NDframe, rows: List[ParsedRow]):
    section("15. StringAccessor idx_* 索引")
    texts = [r["text"] for r in rows]
    r_hello = re.compile(r"hello")
    r_macau = re.compile(r"Macau")

    check("idx_contains(hello)", df.text.idx_contains(r_hello), [i for i, t in enumerate(texts) if r_hello.search(t)])
    check("idx_contains(Macau)", df.text.idx_contains(r_macau), [i for i, t in enumerate(texts) if r_macau.search(t)])

    # strlen idx
    lengths = [len(t) for t in texts]
    check("idx_strlen_greater(10)", df.text.idx_strlen_greater(10), [i for i, l in enumerate(lengths) if l > 10])
    check("idx_strlen_less(10)", df.text.idx_strlen_less(10), [i for i, l in enumerate(lengths) if l < 10])
    check("idx_strlen_equal(15)", df.text.idx_strlen_equal(15), [i for i, l in enumerate(lengths) if l == 15])

    # regex + strlen idx combo
    r_all = re.compile(r".")
    check("idx_regex_strlen_greater(10)", df.text.idx_regex_strlen_greater(r_all, 10), [i for i, l in enumerate(lengths) if l > 10])
    check("idx_regex_strlen_less(10)", df.text.idx_regex_strlen_less(r_all, 10), [i for i, l in enumerate(lengths) if l < 10])
    check("idx_regex_strlen_equal(15)", df.text.idx_regex_strlen_equal(r_all, 15), [i for i, l in enumerate(lengths) if l == 15])

    # idx_contains_in_bbox
    r_M = re.compile(r"M")
    check("idx_contains_in_bbox M in wide bbox", len(df.text.idx_contains_in_bbox(r_M, 0, 0, 2000, 2000)) > 0, True)


def test_string_drop_regex(df: NDframe, rows: List[ParsedRow]):
    section("16. StringAccessor drop_regex")
    texts = [r["text"] for r in rows]
    r_hello = re.compile(r"hello")
    r_nope = re.compile(r"ZZZZNONEXIST")

    df_dr = df.copy()
    df_dr.text.drop_regex(r_hello)
    expected_count = sum(1 for t in texts if not r_hello.search(t))
    check("drop_regex(hello) len", len(df_dr), expected_count)

    df_dr2 = df.copy()
    df_dr2.text.drop_regex(r_nope)
    check("drop_regex(no match) len", len(df_dr2), len(rows))


# ──────────────────────────────────────────────────────────────
# 跨字段替换
# ──────────────────────────────────────────────────────────────

def test_cross_replace(df: NDframe, rows: List[ParsedRow]):
    section("17. 跨字段替换")
    texts = [r["text"] for r in rows]
    seps = [r["text_sep"] for r in rows]

    # replace_to
    df_cr = df.copy()
    df_cr.text.replace_to(" ", "_", "text_sep")
    expected_seps = [t.replace(" ", "_") for t in texts]
    check("replace_to text→text_sep", [df_cr.text_sep.values(i) for i in range(len(rows))], expected_seps)
    check("replace_to text unchanged", list(df_cr.text.values()), texts)

    # regex_replace_to
    df_cr2 = df.copy()
    df_cr2.text.regex_replace_to(re.compile(r"\d+"), "[X]", "text_sep")
    expected = [re.sub(r"\d+", "[X]", t) for t in texts]
    check("regex_replace_to text→text_sep", list(df_cr2.text_sep.values()), expected)


# ──────────────────────────────────────────────────────────────
# 空间过滤 / 索引
# ──────────────────────────────────────────────────────────────

def test_spatial_filter(df: NDframe, rows: List[ParsedRow]):
    section("18. 坐标过滤 filter_in_*")
    # filter_in_bbox(500, 200, 1200, 1000): x0>500 && y0>200 && x1<1200 && y1<1000
    in_bbox = [r for r in rows if r["x0"] > 500 and r["y0"] > 200 and r["x1"] < 1200 and r["y1"] < 1000]
    check("filter_in_bbox(500,200,1200,1000)", len(df.filter_in_bbox(500, 200, 1200, 1000)), len(in_bbox))

    # filter_in_x0x1(500, 1200): x0>500 && x1<1200
    in_x = [r for r in rows if r["x0"] > 500 and r["x1"] < 1200]
    check("filter_in_x0x1(500,1200)", len(df.filter_in_x0x1(500, 1200)), len(in_x))

    # filter_in_y0y1(200, 800): y0>200 && y1<800
    in_y = [r for r in rows if r["y0"] > 200 and r["y1"] < 800]
    check("filter_in_y0y1(200,800)", len(df.filter_in_y0y1(200, 800)), len(in_y))

    # filter_greater_point(500, 200): x0>500 && y0>200
    gp = [r for r in rows if r["x0"] > 500 and r["y0"] > 200]
    check("filter_greater_point(500,200)", len(df.filter_greater_point(500, 200)), len(gp))

    # filter_less_point(1200, 1000): x1<1200 && y1<1000
    lp = [r for r in rows if r["x1"] < 1200 and r["y1"] < 1000]
    check("filter_less_point(1200,1000)", len(df.filter_less_point(1200, 1000)), len(lp))

    # Negative clamp (clamps -1,-5 → 0,0). Choose a bbox wide enough to include rows.
    result = df.filter_in_bbox(-1, -5, 2000, 2000)
    check("filter_in_bbox neg clamp runs without error", len(result) > 0, True)

    # Invalid boundary raises
    try:
        df.filter_in_bbox(20, 0, 10, 10)
        check("filter_in_bbox x0>x1 raises", False, True)
    except ValueError:
        check("filter_in_bbox x0>x1 raises", True, True)


def test_spatial_idx(df: NDframe, rows: List[ParsedRow]):
    section("19. 坐标索引 idx_in_*")
    expected = [i for i, r in enumerate(rows) if r["x0"] > 500 and r["y0"] > 200 and r["x1"] < 1200 and r["y1"] < 1000]
    check("idx_in_bbox(500,200,1200,1000)", df.idx_in_bbox(500, 200, 1200, 1000), expected)

    expected_x = [i for i, r in enumerate(rows) if r["x0"] > 500 and r["x1"] < 1200]
    check("idx_in_x0x1(500,1200)", df.idx_in_x0x1(500, 1200), expected_x)

    expected_y = [i for i, r in enumerate(rows) if r["y0"] > 200 and r["y1"] < 800]
    check("idx_in_y0y1(200,800)", df.idx_in_y0y1(200, 800), expected_y)

    # Validation
    try:
        df.idx_in_bbox(20, 0, 10, 10)
        check("idx_in_bbox x0>x1 raises", False, True)
    except ValueError:
        check("idx_in_bbox x0>x1 raises", True, True)


# ──────────────────────────────────────────────────────────────
# 删除 / 合并 / df_from_idx
# ──────────────────────────────────────────────────────────────

def test_drop_operations(df: NDframe, rows: List[ParsedRow]):
    section("20. 删除操作 drop / drop_in_bbox / df_drop / df_drops / drop_by_idx")

    # drop (in-place)
    df_d1 = df.copy()
    df_d1.drop(0)
    check("drop(0) len", len(df_d1), len(rows) - 1)
    check("drop(0) first text", df_d1[0]["text"], rows[1]["text"])

    # df_drop (returns new)
    df_dd = df.copy()
    df_dd_new = df_dd.df_drop(0)
    check("df_drop len", len(df_dd_new), len(rows) - 1)
    check("df_drop original unchanged", len(df_dd), len(rows))

    # df_drops (returns new)
    df_dds = df.copy()
    df_dds_new = df_dds.df_drops([0, 2, 4])
    expected_cnt = len(rows) - 3
    check("df_drops len", len(df_dds_new), expected_cnt)
    check("df_drops[0] text", df_dds_new[0]["text"], rows[1]["text"])

    # drop_by_idx (in-place)
    df_dbi = df.copy()
    df_dbi.drop_by_idx([1, 3, 5])
    expected = [i for i in range(len(rows)) if i not in (1, 3, 5)]
    check("drop_by_idx len", len(df_dbi), len(expected))
    if df_dbi:
        check("drop_by_idx[0] text", df_dbi[0]["text"], rows[expected[0]]["text"])

    # df_from_idx
    df_fi = df.df_from_idx([0, 3, 7])
    check("df_from_idx len", len(df_fi), 3)
    check("df_from_idx[0] text", df_fi[0]["text"], rows[0]["text"])
    check("df_from_idx[1] text", df_fi[1]["text"], rows[3]["text"])
    check("df_from_idx[2] text", df_fi[2]["text"], rows[7]["text"])
    check("df_from_idx empty", len(df.df_from_idx([])), 0)
    df_fid = df.df_from_idx([0, 0, 1])
    check("df_from_idx dup", len(df_fid), 3)

    # pop_back
    df_pb = df.copy()
    df_pb.pop_back()
    check("pop_back len", len(df_pb), len(rows) - 1)
    check("pop_back[-1] text", df_pb[-1]["text"], rows[-2]["text"])
    empty_pb = NDframe()
    empty_pb.pop_back()
    check("pop_back empty no crash", len(empty_pb), 0)


def test_copy_head_tail(df: NDframe, rows: List[ParsedRow]):
    section("21. copy / head / tail")
    check("head(3) len", len(df.head(3)), 3)
    check("tail(3) len", len(df.tail(3)), 3)
    head1 = df.head(1)
    check("head(1)[0] text", head1[0]["text"], rows[0]["text"])
    tail1 = df.tail(1)
    check("tail(1)[0] text", tail1[0]["text"], rows[-1]["text"])

    df_copy = df.copy()
    df_copy.x0.add(1000)
    check("copy independent", list(df.x0.values()) != list(df_copy.x0.values()), True)


def test_merge_bbox(df: NDframe, rows: List[ParsedRow]):
    section("22. bbox / centroid / centroid_line / merge_line")
    r0 = rows[0]
    check("bbox(0)", df.bbox(0), (r0["x0"], r0["y0"], r0["x1"], r0["y1"]))
    check("bbox(-1)", df.bbox(-1), (rows[-1]["x0"], rows[-1]["y0"], rows[-1]["x1"], rows[-1]["y1"]))

    # centroid
    cx = sum((r["x0"] + r["x1"]) * 0.5 for r in rows) / len(rows)
    cy = sum((r["y0"] + r["y1"]) * 0.5 for r in rows) / len(rows)
    cxc, cyc = df.centroid()
    check("centroid x", abs(cxc - cx) < 1e-10, True)
    check("centroid y", abs(cyc - cy) < 1e-10, True)

    # centroid_line
    check("centroid_line(0)", df.centroid_line(0), ((r0["x0"] + r0["x1"]) * 0.5, (r0["y0"] + r0["y1"]) * 0.5))

    empty_df = NDframe()
    check("centroid empty", empty_df.centroid(), (0.0, 0.0))

    # merge_line (small test)
    df_ml = NDframe()
    df_ml.append(0, 10, 10, 50, 40, 40, 30, 1, 0, "first", "")
    df_ml.append(1, 52, 12, 100, 42, 48, 30, 1, 0, "second", "")
    df_ml.append(2, 200, 200, 300, 250, 100, 50, 1, 0, "far", "")
    merged = df_ml.merge_line(10, 10, " ")
    check("merge_line len", len(merged), 2)
    check("merge_line[0] text", merged[0]["text"], "first second")


def test_misc_operations(df: NDframe, rows: List[ParsedRow]):
    section("23. reset_index / add_y0y1 / concat / from_dict")

    # reset_index
    df_ri = df.copy()
    df_ri.sort_by_x0(descending=True)
    df_ri.reset_index()
    check("reset_index[0] idx", df_ri[0]["idx"], 0)
    check("reset_index[-1] idx", df_ri[-1]["idx"], len(rows) - 1)

    # add_y0y1
    df_ay = df.copy()
    orig_y0 = list(df_ay.y0.values())
    df_ay.add_y0y1(10)
    check("add_y0y1 y0", list(df_ay.y0.values()), [v + 10 for v in orig_y0])

    # concat
    df_cat = NDframe.concat(df, df.head(2))
    check("concat len", len(df_cat), len(rows) + 2)
    check("concat[0] text", df_cat[0]["text"], df[0]["text"])
    check("concat[-1] text", df_cat[-1]["text"], df.head(2)[-1]["text"])

    # from_dict
    fd = NDframe.from_dict({
        "text": ["A", "B", "C"],
        "x0": [1, 2, 3],
        "x1": [1, 2, 3],
        "y0": [4, 5, 6],
        "y1": [4, 5, 6],
        "width": [4, 5, 6],
        "height": [4, 5, 6],
    })
    check("from_dict len", len(fd), 3)
    check("from_dict text[0]", fd.text.values(0), "A")
    check("from_dict x0", list(fd.x0.values()), [1, 2, 3])


def test_conditional_ops(df: NDframe, rows: List[ParsedRow]):
    section("24. 条件运算 apply_if_*")

    # apply_if_greater_add: if x0 > 800 then width += 10
    df_aga = df.copy()
    df_aga.apply_if_greater_add("x0", "width", 800, 10)
    expected_widths = [r["width"] + 10 if r["x0"] > 800 else r["width"] for r in rows]
    check("apply_if_greater_add x0>800 → width+10", list(df_aga.width.values()), expected_widths)

    # apply_if_less_add: if width < 30 then x0 += 5
    df_ala = df.copy()
    df_ala.apply_if_less_add("width", "x0", 30, 5)
    expected_x0 = [r["x0"] + 5 if r["width"] < 30 else r["x0"] for r in rows]
    check("apply_if_less_add width<30 → x0+5", list(df_ala.x0.values()), expected_x0)

    # apply_if_greater_add_y
    df_agay = df.copy()
    df_agay.apply_if_greater_add_y("width", 150, 5)
    expected_y0 = [r["y0"] + 5 if r["width"] > 150 else r["y0"] for r in rows]
    check("apply_if_greater_add_y width>150 → y0+5", list(df_agay.y0.values()), expected_y0)

    # apply_if_less_add_y
    df_alay = df.copy()
    df_alay.apply_if_less_add_y("height", 20, 3)
    expected_y0_2 = [r["y0"] + 3 if r["height"] < 20 else r["y0"] for r in rows]
    check("apply_if_less_add_y height<20 → y0+3", list(df_alay.y0.values()), expected_y0_2)


def test_drop_in_bbox(df: NDframe, rows: List[ParsedRow]):
    section("25. drop_in_bbox")
    df_d = df.copy()
    df_d.drop_in_bbox(0, 0, 300, 300)
    expected = [r for r in rows if not (r["x0"] > 0 and r["y0"] > 0 and r["x1"] < 300 and r["y1"] < 300)]
    check("drop_in_bbox(0,0,300,300) len", len(df_d), len(expected))


def test_small_df_suite():
    """小型人工构造 DF 的专项测试（处理精确边界值的验证）"""
    section("26. 小型 DF 专项测试")
    df = make_small_df()
    rows_data = [
        {"idx": 0, "x0": 10, "y0": 20, "x1": 100, "y1": 60, "width": 90, "height": 40, "text": "hello world 你好世界", "text_sep": "sep_a"},
        {"idx": 1, "x0": 50, "y0": 80, "x1": 200, "y1": 150, "width": 150, "height": 70, "text": "foo bar baz ", "text_sep": "sep_b_connector"},
        {"idx": 2, "x0": 30, "y0": 40, "x1": 80, "y1": 90, "width": 50, "height": 50, "text": "test 123 regex 正则测试", "text_sep": "sep_c"},
        {"idx": 3, "x0": 5, "y0": 5, "x1": 30, "y1": 30, "width": 25, "height": 25, "text": "short 快 短", "text_sep": "sep_d"},
        {"idx": 4, "x0": 60, "y0": 100, "x1": 180, "y1": 200, "width": 120, "height": 100, "text": "another one 这一个", "text_sep": "sep_e"},
    ]
    print("df=", df.text.values())
    subsection("构造与属性")
    check("len", len(df), 5)
    check("empty", df.empty, False)
    lst = df.to_list()
    check("to_list len", len(lst), 5)

    subsection("数值聚合")
    check("x0.max", df.x0.max(), 60)
    check("x0.min", df.x0.min(), 5)
    check("y0.max", df.y0.max(), 100)
    check("y0.min", df.y0.min(), 5)
    check("width.max", df.width.max(), 150)
    check("x0.mean", df.x0.mean(), 31.0)
    check("x0.mode", df.x0.mode(), (5, 1))
    check("x0.diff", df.x0.diff(), [None, 40, -20, -25, 55])
    x0_var1 = sum((v - 31)**2 for v in [10, 50, 30, 5, 60]) / 4
    check("x0.var", round(df.x0.var(1), 10), round(x0_var1, 10))

    subsection("数值过滤")
    check("x0.greater(40)", len(df.x0.greater(40)), 2)
    check("x0.less(40)", len(df.x0.less(40)), 3)
    check("y0.between(10,90)", len(df.y0.between(10, 90)), 3)

    subsection("数值运算")
    df_a = df.copy()
    orig = list(df_a.x0.values())
    df_a.x0.add(5)
    check("x0.add(5)", list(df_a.x0.values()), [v + 5 for v in orig])
    df_a.x0.sub(3)
    check("x0.add(5).sub(3)", list(df_a.x0.values()), [max(v + 2, 0) for v in orig])

    subsection("字符串操作")
    check("text.join('|')", df.text.join(" | "), " | ".join(r["text"] for r in rows_data))
    check("text.len", df.text.len(), [len(r["text"]) for r in rows_data])
    check("text.count('e')", df.text.count('e'), 6)

    # replace
    df_r = df.copy()
    df_r.text.replace(" ", "_")
    check("text.replace", df_r.text.join("|"), "|".join(r["text"].replace(" ", "_") for r in rows_data))

    # strip
    df_st = df.copy()
    df_st.text.strip()
    check("text.strip[1] endswith space", df_st.to_list()[1]["text"].endswith(" "), False)

    # capitalize / swapcase
    df_cap = df.copy()
    df_cap.text.capitalize()
    check("capitalize[0]", df_cap.to_list()[0]["text"][0], "H")
    df_swp = df.copy()
    df_swp.text.swapcase()
    check("swapcase[0]", df_swp.to_list()[0]["text"][:5], "HELLO")

    subsection("字符类型判断")
    df_is = NDframe()
    for t in ["abc123", "hello", "123", "   ", "HELLO", "Hello World"]:
        df_is.append(0, 0, 0, 0, 0, 0, 0, 0, 0, t, "")
    check("isalnum", len(df_is.text.isalnum()), 4)
    check("isalpha", len(df_is.text.isalpha()), 2)
    check("isdigit", len(df_is.text.isdigit()), 1)
    check("isspace", len(df_is.text.isspace()), 1)
    check("islower", len(df_is.text.islower()), 2)
    check("isupper", len(df_is.text.isupper()), 1)
    check("istitle", len(df_is.text.istitle()), 1)
    check("isnumeric", len(df_is.text.isnumeric()), 1)
    check("isdecimal", len(df_is.text.isdecimal()), 1)

    subsection("中文字符操作")
    df_kc = df.copy()
    df_kc.text.keep_chinese()
    check("keep_chinese[0]", df_kc.to_list()[0]["text"], "你好世界")
    check("keep_chinese[1]", df_kc.to_list()[1]["text"], "")
    check("keep_chinese[3]", df_kc.to_list()[3]["text"], "快短")

    df_knp = df.copy()
    df_knp.text.keep_numeric_punct()
    check("keep_numeric_punct[2]", df_knp.to_list()[2]["text"], "123")

    df_iscn = NDframe()
    df_iscn.append(0, 0, 0, 0, 0, 0, 0, 0, 0, "hello世界", "")
    df_iscn.append(1, 0, 0, 0, 0, 0, 0, 0, 0, "你好世界", "")
    check("is_chinese", len(df_iscn.text.is_chinese()), 1)
    check("is_chinese[0] text", df_iscn.text.is_chinese().to_list()[0]["text"], "你好世界")

    subsection("正则")
    df_rr = df.copy()
    df_rr.text.regex_replace(re.compile(r"\d+"), "X")
    check("regex_replace d+ -> X", df_rr.to_list()[2]["text"], "test X regex 正则测试")

    subsection("字符串长度过滤")
    check("strlen_greater(10)", len(df.text.strlen_greater(10)), 4)
    check("strlen_less(10)", len(df.text.strlen_less(10)), 1)
    check("strlen_equal(16)", len(df.text.strlen_equal(16)), 1)

    subsection("集合筛选")
    check("is_in", len(df.text.is_in(["hello world 你好世界", "short 快 短"])), 2)
    check("not_in", len(df.text.not_in(["hello world 你好世界"])), 4)
    check("contains_any True", df.text.contains_any(["hello", "nope"]), True)
    check("contains_any False", df.text.contains_any(["nonexistent"]), False)

    subsection("跨字段替换")
    df_cr = df.copy()
    df_cr.text.replace_to(" ", "_", "text_sep")
    check("replace_to text→text_sep[0]", df_cr.to_list()[0]["text_sep"], "hello_world_你好世界")
    check("replace_to text unchanged", list(df_cr.text.values()), list(df.text.values()))

    df_cr2 = df.copy()
    df_cr2.text_sep.replace_to("sep", "REPLACED", "text")
    check("replace_to sep→text[0]", df_cr2.to_list()[0]["text"], "REPLACED_a")

    subsection("排序")
    df_s = df.copy()
    df_s.x0.sort()
    check("x0.sort", list(df_s.x0.values()), [5, 10, 30, 50, 60])
    df_s2 = df.copy()
    df_s2.y0.sort(descending=True)
    check("y0.sort desc", list(df_s2.y0.values()), [100, 80, 40, 20, 5])

    subsection("索引 / 切片 / 复制")
    check("head(2)", len(df.head(2)), 2)
    check("tail(2)", len(df.tail(2)), 2)
    check("df[0]['text']", df[0]["text"], "hello world 你好世界")
    check("df[-1]['text']", df[-1]["text"], "another one 这一个")
    check("df[1:3] len", len(df[1:3]), 2)

    subsection("drop / add_y0y1")
    df_d = df.copy()
    df_d.drop(0)
    check("drop(0)", len(df_d), 4)
    df_ay = df.copy()
    orig = list(df_ay.y0.values())
    df_ay.add_y0y1(10)
    check("add_y0y1", list(df_ay.y0.values()), [v + 10 for v in orig])

    subsection("merge_line")
    df_ml = NDframe()
    df_ml.append(0, 10, 10, 50, 40, 40, 30, 1, 0, "first", "")
    df_ml.append(1, 52, 12, 100, 42, 48, 30, 1, 0, "second", "")
    df_ml.append(2, 200, 200, 300, 250, 100, 50, 1, 0, "far", "")
    merged = df_ml.merge_line(10, 10, " ")
    check("merge_line len", len(merged), 2)
    check("merge_line text", merged[0]["text"], "first second")

    subsection("sort_by_x1 / sort_by_y1")
    df_sy1 = df.copy()
    df_sy1.sort_by_y1()
    check("sort_by_y1", list(df_sy1.y1.values()), [30, 60, 90, 150, 200])

    subsection("条件运算")
    df_aga = df.copy()
    df_aga.apply_if_greater_add("x0", "width", 20, 10)
    check("apply_if_greater_add", list(df_aga.width.values()), [90, 160, 60, 25, 130])
    df_ala = df.copy()
    df_ala.apply_if_less_add("x0", "width", 20, 5)
    check("apply_if_less_add", list(df_ala.width.values()), [95, 150, 50, 30, 120])

    # drop_in_bbox
    df_dbi = df.copy()
    df_dbi.drop_in_bbox(0, 0, 30, 30)
    check("drop_in_bbox strict", len(df_dbi), 5)

    # bbox / centroid
    check("bbox(0)", df.bbox(0), (10, 20, 100, 60))
    check("centroid_line(0)", df.centroid_line(0), (55.0, 40.0))

    # from_dict
    fd = NDframe.from_dict({"text": ["A", "B"], "x0": [1, 2], "y0": [1, 2],
                            "x1": [1, 2], "y1": [1, 2], "width": [1, 2], "height": [1, 2]})
    check("from_dict", len(fd), 2)

    # concat
    check("concat", len(NDframe.concat(df, df.head(2))), 7)

    # pop_back
    df_pb = df.copy()
    df_pb.pop_back()
    check("pop_back", len(df_pb), 4)

    # drop_by_idx
    df_dbi = df.copy()
    df_dbi.drop_by_idx([1, 3])
    check("drop_by_idx len", len(df_dbi), 3)

    # drop_regex
    df_dr = df.copy()
    df_dr.text.drop_regex(re.compile(r"hello"))
    check("drop_regex", len(df_dr), 4)

    # idx_*
    check("idx_contains hello", df.text.idx_contains(re.compile(r"hello")), [0])
    check("idx_strlen_greater 10", df.text.idx_strlen_greater(10), [0, 1, 2, 4])


# ════════════════════════════════════════════════════════════════
# 性能压力测试
# ════════════════════════════════════════════════════════════════

def _random_text(length: int) -> str:
    """生成长度约 length 的随机中英混合文本"""
    parts = []
    while len(parts) < length:
        choice = random.random()
        if choice < 0.4:
            # ASCII word
            wlen = random.randint(2, 12)
            parts.append("".join(random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? ") for _ in range(wlen)))
        elif choice < 0.7:
            # CJK characters
            parts.append(chr(random.randint(0x4E00, 0x9FFF)))
        else:
            # mixed separator
            parts.append(" ")
    return "".join(parts[:length])


def _random_coord() -> Tuple[int, int, int, int]:
    """随机坐标 (left, top, width, height)"""
    left = random.randint(0, 2000)
    top = random.randint(0, 2000)
    w = random.randint(5, 300)
    h = random.randint(5, 200)
    return left, top, w, h


def _build_random_df(n: int) -> NDframe:
    """构造 n 行随机数据的 NDframe"""
    df = NDframe(n)
    for i in range(n):
        left, top, w, h = _random_coord()
        page = random.randint(1, 5)
        flags = random.randint(0, 3)
        txt = _random_text(random.randint(1, 30))
        df.append(i, left, top, left + w, top + h, w, h, page, flags, txt, "")
    return df


def test_performance_small():
    section("100. 性能测试 — 小规模 (1000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(1000)
    dt = time.perf_counter() - t0
    check("build 1000 rows", len(df), 1000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    for name, acc in [("x0", df.x0), ("y0", df.y0), ("width", df.width), ("height", df.height)]:
        t0 = time.perf_counter()
        _ = acc.max(), acc.min(), acc.mean()
        dt = time.perf_counter() - t0
        print(f"   {name}.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    t0 = time.perf_counter()
    _ = df.x0.greater(500)
    dt = time.perf_counter() - t0
    print(f"   x0.greater(500): {dt*1000:.2f} ms")

    subsection("字符串操作")
    t0 = time.perf_counter()
    str_len_list = df.text.len()
    print("str_len_list:", str_len_list)
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.count("e")
    dt = time.perf_counter() - t0
    print(f"   text.count('e'): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字): {dt*1000:.2f} ms")

    subsection("排序")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("复制/转换")
    t0 = time.perf_counter()
    _ = df.copy()
    dt = time.perf_counter() - t0
    print(f"   copy: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_list()
    dt = time.perf_counter() - t0
    print(f"   to_list: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_string()
    dt = time.perf_counter() - t0
    print(f"   to_string: {dt*1000:.2f} ms")


def test_performance_medium():
    section("101. 性能测试 — 中规模 (10000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(10000)
    dt = time.perf_counter() - t0
    check("build 10000 rows", len(df), 10000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    for name in ("x0", "y0", "width", "height"):
        acc = getattr(df, name)
        t0 = time.perf_counter()
        _ = acc.max(), acc.min(), acc.mean()
        dt = time.perf_counter() - t0
        print(f"   {name}.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    for name in ("x0", "y0"):
        acc = getattr(df, name)
        t0 = time.perf_counter()
        _ = acc.greater(500)
        dt = time.perf_counter() - t0
        print(f"   {name}.greater(500): {dt*1000:.2f} ms")

    subsection("字符串扫描")
    t0 = time.perf_counter()
    _ = df.text.len()
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.text.count("e")
    dt = time.perf_counter() - t0
    print(f"   text.count('e'): {dt*1000:.2f} ms")

    subsection("正则过滤")
    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字) 过滤: {dt*1000:.2f} ms")

    subsection("排序")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("字符串变换")
    t0 = time.perf_counter()
    df_t = df.copy()
    df_t.text.strip()
    dt = time.perf_counter() - t0
    print(f"   text.strip: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    df_t2 = df.copy()
    df_t2.text.replace(" ", "_")
    dt = time.perf_counter() - t0
    print(f"   text.replace(' ', '_'): {dt*1000:.2f} ms")

    subsection("删除操作")
    t0 = time.perf_counter()
    df_d = df.copy()
    df_d.drop_by_idx(list(range(0, 10000, 2)))
    dt = time.perf_counter() - t0
    check(f"drop_by_idx half rows", len(df_d), 5000)
    print(f"   drop_by_idx(half): {dt*1000:.1f} ms")

    subsection("空间过滤")
    t0 = time.perf_counter()
    _ = df.filter_in_bbox(100, 100, 1000, 1000)
    dt = time.perf_counter() - t0
    print(f"   filter_in_bbox: {dt*1000:.2f} ms")

    subsection("concat")
    t0 = time.perf_counter()
    _ = NDframe.concat(df, df)
    dt = time.perf_counter() - t0
    print(f"   concat(self, self): {dt*1000:.2f} ms")

    subsection("from_dict")
    d = {
        "text": ["hello"] * 10000,
        "x0": [i for i in range(10000)],
        "x1": [i for i in range(10000)],
        "y0": [i * 2 for i in range(10000)],
        "y1": [i * 2 for i in range(10000)],
        "width": [50] * 10000,
        "height": [30] * 10000,
    }
    t0 = time.perf_counter()
    _ = NDframe.from_dict(d)
    dt = time.perf_counter() - t0
    print(f"   from_dict(10000): {dt*1000:.2f} ms")


def test_performance_large():
    section("102. 性能测试 — 大规模 (100000 rows)")
    subsection("构造")
    t0 = time.perf_counter()
    df = _build_random_df(100000)
    dt = time.perf_counter() - t0
    check("build 100000 rows", len(df), 100000)
    print(f"   构建耗时: {dt*1000:.1f} ms")

    subsection("数值聚合")
    t0 = time.perf_counter()
    _ = df.x0.max(), df.x0.min(), df.x0.mean()
    dt = time.perf_counter() - t0
    print(f"   x0.max/min/mean: {dt*1000:.2f} ms")

    subsection("数值过滤")
    t0 = time.perf_counter()
    _ = df.x0.greater(500)
    dt = time.perf_counter() - t0
    print(f"   x0.greater(500): {dt*1000:.2f} ms")

    subsection("字符串扫描")
    t0 = time.perf_counter()
    _ = df.text.len()
    dt = time.perf_counter() - t0
    print(f"   text.len(): {dt*1000:.2f} ms")

    subsection("排序 (100k)")
    t0 = time.perf_counter()
    df_s = df.copy()
    df_s.x0.sort()
    dt = time.perf_counter() - t0
    print(f"   x0.sort: {dt*1000:.2f} ms")

    subsection("正则过滤 100k")
    t0 = time.perf_counter()
    _ = df.text.contains(re.compile(r"\d+"))
    dt = time.perf_counter() - t0
    print(f"   text.contains(数字): {dt*1000:.2f} ms")

    subsection("字符串替换 100k")
    t0 = time.perf_counter()
    df_t = df.copy()
    df_t.text.replace(" ", "_")
    dt = time.perf_counter() - t0
    print(f"   text.replace(' ', '_'): {dt*1000:.2f} ms")

    subsection("复制 / to_list")
    t0 = time.perf_counter()
    _ = df.copy()
    dt = time.perf_counter() - t0
    print(f"   copy: {dt*1000:.2f} ms")

    t0 = time.perf_counter()
    _ = df.to_list()
    dt = time.perf_counter() - t0
    print(f"   to_list: {dt*1000:.2f} ms")


# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  ndframe 完整功能测试套件")
    print("=" * 60)

    # 加载真实 OCR 数据（86 行）
    df, rows = load_test_data()
    df.print()
    print("df=", df.text.values())
    print("df=", df.text.len())
    print("df=", Counter(df.text.len()))
    print(f"\n加载测试数据: {len(rows)} 行")

    # ── 功能测试（使用 86 行真实数据）──
    test_basic_properties(df, rows)
    test_to_list_string(df, rows)
    test_numeric_aggregation(df, rows)
    test_numeric_filter(df, rows)
    test_numeric_arithmetic(df, rows)
    test_numeric_sort(df, rows)
    test_numeric_idx(df, rows)
    test_string_basic(df, rows)
    test_string_transform(df, rows)
    test_string_is(df, rows)
    test_string_starts_ends(df, rows)
    test_string_regex(df, rows)
    test_string_len_filter(df, rows)
    test_string_set_ops(df, rows)
    test_string_idx_ops(df, rows)
    test_string_drop_regex(df, rows)
    test_cross_replace(df, rows)
    test_spatial_filter(df, rows)
    test_spatial_idx(df, rows)
    test_drop_operations(df, rows)
    test_copy_head_tail(df, rows)
    test_merge_bbox(df, rows)
    test_misc_operations(df, rows)
    test_conditional_ops(df, rows)
    test_drop_in_bbox(df, rows)

    # ── 小型人工数据精确测试 ──
    test_small_df_suite()

    # ── 性能压力测试 ──
    test_performance_small()
    test_performance_medium()
    test_performance_large()

    # ── 汇总 ──
    total = PASS + FAIL
    print(f"\n{'=' * 60}")
    print(f"  结果: {PASS}/{total} 通过, {FAIL} 失败")
    print(f"{'=' * 60}")

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


    """
    # 1. 区域限制 + 正则过滤的完美交集
    # 找出“既在表格内容区域(x0>100, y0>200...)内，又包含金额数字”的行：
    df_table_area = df.filter_in_bbox(100, 200, 800, 900)
    df_money_lines = df.text.contains(r"\d+\.\d{2}")
    
    # 一行搞定交集运算，耗时约 5~10 微秒：
    df_target = df_table_area & df_money_lines
    
    # 2. 白名单 idx 快速过滤：
    df_selected = df & [0, 4, 12, 19]
    
    # 3. 甚至可以和 +、- 组合使用复杂的集合表达式：
    # (区域 A 和 区域 B 的并集) 中排除 包含数字的行
    result = (df_area_a + df_area_b) - df.text.contains(r"\d+")
    """