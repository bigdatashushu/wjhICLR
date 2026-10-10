"""类别名匹配（v6 §9.1/§9.2 的对象查询口径）。

**为什么需要单独一个模块**：VSI-Bench 的题面用自然语言点名物体
（"the telephone"、"trash can"、"blue chair"），而 M5 的清单里是检测器给的
类别名（`phone`、`trash_can`、`chair`）。两者之间有三类系统性差异，任何一类
没处理都会让**整类题型**因为"找不到对象"而 abstain：

1. **分隔符差异**：`trash_can` vs `trash can`、`coffee-table` vs `coffee table`；
2. **同义词差异**：`telephone` vs `phone`、`sofa` vs `couch`、`tv` vs `television`；
3. **修饰词/单复数**：`blue chair` vs `chair`、`computer mouse` vs `mouse`、
   `books` vs `book`。

2026-09-21 真实实测（32 题 inner_validation，scene 7b6477cb95）：清单里有
`phone`（1 个）与 `trash_can`（1 个），而题面写的是 `telephone` 与 `trash can` →
`list_objects('telephone')` 与 `list_objects('trash can')` **都返回空** →
程序走 `if not objs: ReturnAnswer("abstain")` → 4 个 rel_direction 题全 abstain。
这不是几何/证据的问题，纯粹是字符串匹配没做归一。

匹配规则（确定性、可解释）：
- 归一：小写、`_-/` → 空格、压空格、去首尾；
- 单复数：简单词尾剥离（`s` / `es`）后比较（不做词干递归）；
- 同义词：`SYNONYMS` 表把常见室内同义词折叠到同一规范名；
- 命中判据：移除登记的外观前缀后，规范类别必须相等；
  不以子串或任意共享单词判定类别。`bookshelf`、`book`、`shelf` 各自独立。
  外观修饰只用于类别检索，具体实例仍由 SceneHandle 做唯一性检查。
"""

from __future__ import annotations

import re

# 同义词 → 规范名（只收室内场景常见且**语义等价**的；不做上下位关系推断）
SYNONYMS: dict[str, str] = {
    "telephone": "phone",
    "cellphone": "phone",
    "mobile": "phone",
    "smartphone": "phone",
    "landline": "phone",
    "trash": "trash can",
    "garbage": "trash can",
    "bin": "trash can",
    "wastebasket": "trash can",
    "waste basket": "trash can",
    "rubbish bin": "trash can",
    "sofa": "couch",
    "settee": "couch",
    "tv": "television",
    "television": "television",
    "monitor": "monitor",
    "screen": "monitor",
    "display": "monitor",
    "computer mouse": "mouse",
    "desk chair": "chair",
    "office chair": "chair",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "fridge": "refrigerator",
    "freezer": "refrigerator",
    "nightstand": "nightstand",
    "night stand": "nightstand",
    "bedside table": "nightstand",
    "dresser": "dresser",
    "chest of drawers": "dresser",
    "wardrobe": "wardrobe",
    "closet": "wardrobe",
    "armoire": "wardrobe",
    "stool": "stool",
    "ottoman": "ottoman",
    "footstool": "ottoman",
    "rug": "rug",
    "carpet": "rug",
    "mat": "rug",
    "picture": "picture",
    "painting": "picture",
    "artwork": "picture",
    "poster": "picture",
    "lamp": "lamp",
    "light": "lamp",
    "ceiling light": "ceiling lamp",
    "chandelier": "ceiling lamp",
    "countertop": "counter",
    "counter top": "counter",
    "desk counter": "desk",
    "cutting board": "cutting board",
    "chopping board": "cutting board",
    "pan": "pan",
    "pot": "pot",
    "kettle": "kettle",
    "microwave": "microwave",
    "oven": "oven",
    "stove": "stove",
    "sink": "sink",
    "cupboard": "cabinet",
    "cabinet": "cabinet",
    "shelf": "shelf",
    "bookshelf": "bookshelf",
    "bookcase": "bookshelf",
    "plant": "plant",
    "flowerpot": "plant",
    "vase": "vase",
    "bottle": "bottle",
    "bowl": "bowl",
    "cup": "cup",
    "mug": "cup",
    "glass": "cup",
    "tray": "tray",
    "box": "box",
    "basket": "basket",
    "bag": "bag",
    "backpack": "bag",
    "towel": "towel",
    "blanket": "blanket",
    "pillow": "pillow",
    "cushion": "pillow",
    "curtain": "curtain",
    "blinds": "blinds",
    "door": "door",
    "doorway": "door",
    "window": "window",
    "mirror": "mirror",
    "whiteboard": "whiteboard",
    "blackboard": "whiteboard",
    "fan": "fan",
    "heater": "heater",
    "radiator": "radiator",
    "air conditioner": "air conditioner",
    "printer": "printer",
    "projector": "projector",
    "speaker": "speaker",
    "clock": "clock",
    "telephone handset": "phone",
    # ---- v7 新增（2026-09-22，真实实测）：VSI-Bench 高频点名但旧表缺失的类别。
    # 这些词在 inner 档题面里反复出现；缺表会让检测器标签与题面用词对不上，
    # 表现为"题面点名的物体解析失败"（实测 16 道 rel_direction 里 13 道因此作废）。
    "power strip": "power strip",
    "extension cord": "power strip",
    "power outlet": "power strip",
    "computer tower": "computer tower",
    "pc tower": "computer tower",
    "desktop computer": "computer tower",
    "coat rack": "coat rack",
    "paper bag": "paper bag",
    "shopping bag": "paper bag",
    "exhaust fan": "fan",
    "ceiling fan": "fan",
    "suitcase": "suitcase",
    "luggage": "suitcase",
    "bucket": "bucket",
    "pail": "bucket",
    "column": "column",
    "pillar": "column",
    "laptop": "laptop",
    "computer": "computer",
    "desk lamp": "lamp",
    "floor lamp": "lamp",
    "table lamp": "lamp",
    "water bottle": "water bottle",
    "coffee maker": "coffee maker",
    "coffee machine": "coffee maker",
    "doorknob": "doorknob",
    "door knob": "doorknob",
    "drawer": "drawer",
    "camera": "camera",
    "hanger": "hanger",
    "shower": "shower",
    "wall": "wall",
    "board": "board",
    "pen": "pen",
    "tray": "tray",
}

_WS = re.compile(r"[\s_\-/,]+")
_TOKEN = re.compile(r"[a-z0-9]+")
CATEGORY_MATCH_VERSION = "category-match-v11.2"
_APPEARANCE_PREFIXES = frozenset({
    "the", "a", "an", "black", "white", "blue", "red", "green", "yellow",
    "orange", "brown", "gray", "grey", "pink", "purple", "beige",
    "small", "large", "big", "wooden", "metal", "plastic",
})


def normalize(name: str) -> str:
    """归一化类别名：小写、分隔符折叠、压空格、去首尾。"""
    s = str(name or "").strip().lower()
    s = _WS.sub(" ", s)
    return s.strip()


def canonical(name: str) -> str:
    """归一化 + 单复数折叠 + 同义词折叠 → 规范名。

    **顺序很关键**：必须先折叠单复数再查同义词表，否则 `Sofas`（表里只有
    `sofa`）与 `telephones`（表里只有 `telephone`）都会漏掉 —— 且 `telephones`
    会因 `es` 剥离变成 `telephon`，比不折叠更糟。这是实测踩过的顺序 bug。
    同义词折叠做**一趟**（折叠结果若仍是表里的键再查一次），不递归。
    """
    s = normalize(name)
    if not s:
        return ""
    s = _depluralize(s)
    s = SYNONYMS.get(s, s)
    return SYNONYMS.get(s, s)


def _depluralize(s: str) -> str:
    """简单单复数折叠（不做词干递归）：`books`→`book`、`boxes`→`box`。

    `es` 只在**词干以 s/x/z/ch/sh 结尾**时剥离（`boxes`→`box`、`churches`→`church`）；
    否则只剥 `s`。这条区分是必须的：对 `telephones` 直接剥 `es` 会得到 `telephon`，
    比不折叠更糟（实测踩过）。
    """
    out = []
    for w in s.split():
        if len(w) > 4 and w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
            w = w[:-2]
        elif len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def tokens(name: str) -> set[str]:
    """规范名的单词集合（供调用方分析，不作为类别命中判据）。"""
    return set(_TOKEN.findall(canonical(name)))


def option_category(value: str) -> str:
    """对象选项去掉可选字母前缀；只读取题面选项，不读取答案标签。"""
    return re.sub(r"^\s*(?:\([A-Za-z]\)|[A-Za-z][.)])\s*", "", str(value)).strip()


def matches(query: str, category_name: str) -> bool:
    """`query`（题面用词）是否指向 `category_name`（清单里的类别）。

    只去掉明确的外观前缀，再做完整类别与同义词匹配；不猜复合名词的含义。
    空查询/空类别名 → False（不把"没写类别"当成"匹配一切"）。
    """
    def category(value: str) -> str:
        words = normalize(value).split()
        while len(words) > 1 and words[0] in _APPEARANCE_PREFIXES:
            words.pop(0)
        return canonical(" ".join(words))

    q, c = category(query), category(category_name)
    return bool(q and c and q == c)


__all__ = ["CATEGORY_MATCH_VERSION", "SYNONYMS", "canonical", "matches",
           "normalize", "option_category", "tokens"]
