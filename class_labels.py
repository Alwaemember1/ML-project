"""仅用于展示的明确名称映射，不改变检查点类别名称或编号。

不确定的专有病名保留原始英文。39 类包含健康、病害/虫害、背景。
"""

CHINESE_LABELS = {
    "Apple___Apple_scab": "苹果 · 黑星病",
    "Apple___Black_rot": "苹果 · 黑腐病",
    "Apple___Cedar_apple_rust": "苹果 · 雪松苹果锈病",
    "Apple___healthy": "苹果 · 健康",
    "Background_without_leaves": "无叶片背景",
    "Blueberry___healthy": "蓝莓 · 健康",
    "Cherry___Powdery_mildew": "樱桃 · 白粉病",
    "Cherry___healthy": "樱桃 · 健康",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "玉米 · 灰斑病",
    "Corn___Common_rust": "玉米 · 普通锈病",
    "Corn___Northern_Leaf_Blight": "玉米 · 大斑病",
    "Corn___healthy": "玉米 · 健康",
    "Grape___Black_rot": "葡萄 · 黑腐病",
    "Grape___Esca_(Black_Measles)": "葡萄 · 黑麻疹病（Esca）",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄 · 叶枯病（Isariopsis 叶斑病）",
    "Grape___healthy": "葡萄 · 健康",
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘 · 黄龙病",
    "Peach___Bacterial_spot": "桃 · 细菌性斑点病",
    "Peach___healthy": "桃 · 健康",
    "Pepper,_bell___Bacterial_spot": "甜椒 · 细菌性斑点病",
    "Pepper,_bell___healthy": "甜椒 · 健康",
    "Potato___Early_blight": "马铃薯 · 早疫病",
    "Potato___Late_blight": "马铃薯 · 晚疫病",
    "Potato___healthy": "马铃薯 · 健康",
    "Raspberry___healthy": "树莓 · 健康",
    "Soybean___healthy": "大豆 · 健康",
    "Squash___Powdery_mildew": "南瓜 · 白粉病",
    "Strawberry___Leaf_scorch": "草莓 · 叶焦病",
    "Strawberry___healthy": "草莓 · 健康",
    "Tomato___Bacterial_spot": "番茄 · 细菌性斑点病",
    "Tomato___Early_blight": "番茄 · 早疫病",
    "Tomato___Late_blight": "番茄 · 晚疫病",
    "Tomato___Leaf_Mold": "番茄 · 叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄 · Septoria 叶斑病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄 · 二斑叶螨危害",
    "Tomato___Target_Spot": "番茄 · 靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄 · 黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus": "番茄 · 花叶病毒病",
    "Tomato___healthy": "番茄 · 健康",
}


def display_name(raw_name):
    return CHINESE_LABELS.get(raw_name, raw_name)


def category_kind(raw_name):
    if raw_name == "Background_without_leaves":
        return "背景类"
    if raw_name.endswith("___healthy"):
        return "健康类"
    return "病害 / 虫害类"
