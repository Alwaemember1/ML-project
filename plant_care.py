"""类别规则参考与防治资料；不是图像严重程度模型，不使用预测概率分级。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from class_labels import CHINESE_LABELS, category_kind, display_name

REFERENCE_VERSION = "category_reference_v1_20261018"
CHECKED_DATE = "2026-10-18"
RISK_SCOPE = "参考风险按预测类别的传播与管理特点设置，是课程演示规则，尚未经田间验证；不是病斑严重程度、发生概率或官方检疫等级。"
SEVERITY_SCOPE = "未评估：现有数据没有病斑面积、严重程度或田间发病率标注，当前分类模型不能输出真实严重程度。"
USE_SCOPE = "以下资料对应预测类别；先由植保人员复核作物和病因，预测错误时建议不适用。国外资料仅参考通用植保原则，实际用药应依据当地登记标签与植保指导。"

# 资料均为人工核对的政府/大学推广机构来源，运行时不联网。
SOURCES = {
    "ipm": ("UC IPM：综合病虫害管理", "https://ipm.ucanr.edu/what-is-ipm/"),
    "apple_scab": ("UC IPM：苹果和梨黑星病", "https://ipm.ucanr.edu/home-and-landscape/apple-and-pear-scab/"),
    "apple_rot": ("Cooperative Extension：苹果黑腐病", "https://apples.extension.org/black-rot-of-apple/"),
    "apple_rust": ("明尼苏达大学：雪松苹果锈病", "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/cedar-apple-rust"),
    "cherry_mildew": ("UC IPM：樱桃白粉病", "https://ipm.ucanr.edu/agriculture/cherry/powdery-mildew/"),
    "fruit_mildew": ("UC IPM：果树和浆果白粉病", "https://ipm.ucanr.edu/home-and-landscape/powdery-mildew-on-fruits-and-berries/"),
    "corn": ("明尼苏达大学：玉米灰斑病和大斑病管理", "https://blog-crop-news.extension.umn.edu/2024/04/strategic-farming-lets-talk-crops.html"),
    "corn_rust": ("明尼苏达大学：玉米普通锈病", "https://extension.umn.edu/agriculture/crop-production/corn/common-rust-on-corn"),
    "grape_rot": ("马里兰大学：葡萄黑腐病与栽培", "https://www.extension.umd.edu/resource/growing-grapes-home-garden"),
    "grape_esca": ("UC IPM：葡萄 Esca 黑麻疹病", "https://ipm.ucanr.edu/agriculture/grape/esca-black-measles/"),
    "grape_trunk": ("UC IPM：葡萄枝干病管理", "https://ipm.ucanr.edu/agriculture/grape/eutypa-dieback/"),
    "grape_spot": ("Embrapa：葡萄 Isariopsis 叶斑病资料", "https://www.infoteca.cnptia.embrapa.br/infoteca/handle/doc/1060452?mode=full"),
    "hlb": ("农业部：黄龙病健康种苗、木虱与病株综合防控", "https://zzys.moa.gov.cn/gzdt/201609/t20160905_6310134.htm"),
    "peach": ("北卡州立大学：2026年桃与油桃病虫害指南", "https://content.ces.ncsu.edu/north-carolina-peach-and-nectarine-disease-and-pest-management-guide"),
    "pepper": ("UC IPM：甜椒细菌性斑点病", "https://ipm.ucanr.edu/agriculture/peppers/bacterial-spot/"),
    "potato_early": ("UC IPM：马铃薯早疫病", "https://ipm.ucanr.edu/agriculture/potato/early-blight/"),
    "potato_late": ("UC IPM：马铃薯晚疫病", "https://ipm.ucanr.edu/agriculture/potato/late-blight/"),
    "squash": ("UC IPM：葫芦科白粉病", "https://ipm.ucanr.edu/home-and-landscape/powdery-mildew-on-cucurbits/"),
    "strawberry": ("北卡州立大学：草莓叶焦病", "https://content.ces.ncsu.edu/leaf-scorch-of-strawberry"),
    "tomato_spot": ("明尼苏达大学：番茄叶斑病管理", "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/yard-and-garden-problems/tomato-leaf-spot-diseases"),
    "tomato_early": ("UC IPM：番茄早疫病", "https://ipm.ucanr.edu/agriculture/tomato/early-blight/"),
    "tomato_late": ("UC IPM：番茄晚疫病", "https://ipm.ucanr.edu/agriculture/tomato/late-blight/"),
    "tomato_mold": ("明尼苏达大学：番茄叶霉病", "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-leaf-mold"),
    "septoria": ("马里兰大学：番茄 Septoria 叶斑病", "https://www.extension.umd.edu/resource/septoria-leaf-spot-tomatoes"),
    "mites": ("UC IPM：叶螨管理", "https://ipm.ucanr.edu/home-and-landscape/spider-mites/"),
    "target": ("佛罗里达大学 IFAS：番茄靶斑病", "https://ask.ifas.ufl.edu/publication/PP351"),
    "tylcv": ("UC IPM：番茄黄化曲叶病毒病", "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/"),
    "mosaic": ("UC IPM：烟草/番茄花叶病毒", "https://ipm.ucanr.edu/agriculture/tomato/tobacco-mosaic/"),
    "viruses": ("明尼苏达大学：番茄病毒及防传播", "https://extension.umn.edu/agriculture/specialty-crops/vegetable-farming/disease-management/tomato-viruses"),
}


def entry(reason, signs, actions, sources, risk="中"):
    return {"risk_level": risk, "reason": reason, "signs_to_check": signs,
            "actions": actions, "source_ids": sources}


DISEASE_CARE = {
    "Apple___Apple_scab": entry("潮湿条件下可能持续侵染，需核查落叶与果实情况。",
        "核查叶片与果实的暗橄榄色斑，注意与其他叶斑病区分。",
        ["清理或规范处理落叶，减少越冬菌源。", "改善树冠通风，减少叶片长时间湿润；新植时考虑抗病品种。"], ["apple_scab"]),
    "Apple___Black_rot": entry("应结合枝干和僵果复核，叶片照片不足以确认全树病情。",
        "同时观察叶斑、枝干溃疡及树上残留僵果。",
        ["确诊后清除僵果、枯死或患病木质残体。", "巡查枝干受伤和衰弱部位，由果树技术人员确定修剪与防护措施。"], ["apple_rot"]),
    "Apple___Cedar_apple_rust": entry("需结合品种敏感性和转主寄主调查，不应仅凭叶斑决定施药。",
        "观察橙黄色叶斑及叶背突起，并调查附近杜松等转主寄主。",
        ["新植时考虑抗病苹果品种与转主寄主布局。", "由园艺人员调查转主寄主病瘿，避免仅凭一次图片预测移除树木。"], ["apple_rust"]),
    "Cherry___Powdery_mildew": entry("新梢、叶片和果实均需监测，尤其注意新生组织。",
        "复核叶面、嫩梢或果面是否有白色粉状生长物。",
        ["重点巡查树冠内新梢和萌蘖，并记录症状变化。", "结合品种和生育期由植保人员判断防治时机，采用适合当地的果园管理。"], ["cherry_mildew", "fruit_mildew"]),
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": entry("病害发展受湿度和品种影响，需田间调查后判断损失风险。",
        "核查是否为受叶脉限制的长方形灰褐色病斑。",
        ["优先选择适合当地的抗病杂交种。", "记录田间发病分布及天气，避免依据单张叶图决定全田施药。"], ["corn"]),
    "Corn___Common_rust": entry("普通锈病与南方锈病需区分，抗性品种和发生情况影响处置。",
        "观察叶片两面的锈色孢子堆，并请技术人员与其他锈病鉴别。",
        ["优先采用抗病杂交种。", "巡查新发孢子堆和发病范围，必要防治须以田间调查为依据。"], ["corn_rust"]),
    "Corn___Northern_Leaf_Blight": entry("适宜湿润天气可能促进扩展，需结合田间病情复核。",
        "检查是否存在较长的梭形或雪茄形病斑。",
        ["结合当地试验选择抗病品种。", "重点巡查湿润天气后的病斑扩展，记录发病叶位与范围并咨询植保人员。"], ["corn"]),
    "Grape___Black_rot": entry("果实感染与残留僵果相关，叶片之外还需检查果穗。",
        "观察叶片圆形褐斑、细小黑点及果穗上的僵果。",
        ["确诊后清理病果与僵果，减少残留菌源。", "合理整枝改善果穗附近的通风，结合生育期开展田间监测。"], ["grape_rot"]),
    "Grape___Esca_(Black_Measles)": entry("属于复杂枝干病问题，需结合枝干检查，不能承诺喷药治愈。",
        "复核叶脉间条纹，同时由专业人员检查枝干与果实症状。",
        ["标记疑似病株，送诊并调查木质部，避免仅据图片实施大幅修剪。", "确诊后由技术人员制定病枝清理与修剪伤口预防方案。"], ["grape_esca", "grape_trunk"]),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": entry("需复核病原及落叶情况，避免与其他葡萄叶斑病混淆。",
        "检查叶片斑点与提前落叶，确认是否与 Isariopsis 类叶斑一致。",
        ["记录采收前后叶片保留与落叶情况，必要时送检鉴别。", "由当地葡萄植保人员依据确诊结果安排后续叶片保护，不直接照搬国外药剂方案。"], ["grape_spot"]),
    "Orange___Haunglongbing_(Citrus_greening)": entry("疑似黄龙病需要优先复核并管理传播风险，图片不能替代检测。",
        "黄化也可能源自缺素等原因，应送检并检查柑橘木虱。",
        ["暂缓调运疑似植株和繁殖材料，联系当地植保机构复核。", "依据确诊和当地指导，综合采用健康种苗、木虱管理及病株处理；不能只凭模型砍树。"], ["hlb"], "高"),
    "Peach___Bacterial_spot": entry("发生程度受品种、湿润条件和生育期影响，需区分其他斑点或药害。",
        "结合叶片、枝梢及果面斑点进行复核。",
        ["优先选用适合当地、具有抗性的品种。", "记录降雨和发病时间；细菌性问题应确诊后制定方案，不笼统当作真菌病处理。"], ["peach"]),
    "Pepper,_bell___Bacterial_spot": entry("病原可随种子或苗传播，湿润叶面促进发生。",
        "复核叶背水渍状或褐色小斑及果实病斑。",
        ["使用可靠来源的健康种子和无病苗。", "采用滴灌等方式减少叶面淋水，结合当地条件轮作。"], ["pepper"]),
    "Potato___Early_blight": entry("衰弱或老叶需重点监测，管理应结合植株生长状态。",
        "观察老叶同心轮纹斑，同时检查种植管理及块茎症状。",
        ["合理供水施肥，减少植株胁迫。", "开展常规巡查并记录病情，由植保人员判断是否具有需要防治的损失风险。"], ["potato_early"]),
    "Potato___Late_blight": entry("适宜潮湿条件下可快速扩展，疑似结果应优先复核。",
        "检查水渍状不规则病斑、叶背白色生长物及块茎异常。",
        ["尽快联系当地植保人员核查，并调查附近马铃薯及番茄。", "使用可靠健康种薯，清理自生苗和废弃薯堆，改善叶片干燥条件。"], ["potato_late"], "高"),
    "Squash___Powdery_mildew": entry("需要持续监测叶片，白粉病并非必须有叶面水膜才发生。",
        "观察叶面粉状斑及黄化，注意与霜霉病等区分。",
        ["选用适合当地的抗病品种，做好田间卫生与杂草管理。", "从营养生长期到结果期持续巡查，避免过密、过量施肥等不利管理。"], ["squash"]),
    "Strawberry___Leaf_scorch": entry("对产量的影响依生产制度而异，不能只凭叶色判为严重。",
        "复核紫色小斑、焦枯样叶片，并区分白粉病等。",
        ["保持合理株距、通风与排水，控制杂草和过长叶面湿润时间。", "结合生产制度和采后管理清理病残组织，咨询当地技术人员确定必要措施。"], ["strawberry"]),
    "Tomato___Bacterial_spot": entry("可与其他叶斑相似，应先复核病因，再决定针对性处置。",
        "结合叶斑及果面症状鉴别细菌性斑点、斑疹与其他叶斑病。",
        ["使用健康种苗，不从患病植株留种。", "减少叶面淋水和土壤溅水，改善株间通风，及时记录并处理确诊病残体。"], ["tomato_spot"]),
    "Tomato___Early_blight": entry("通常需结合老叶、茎和果实症状复核，易与其他叶斑混淆。",
        "检查老叶轮纹斑及相邻茎、果实，避免只看单一叶斑。",
        ["减少喷灌与土壤溅水，采用适当覆盖、支架和通风管理。", "调查并管理田间自生番茄、马铃薯及相关杂草，清理确诊病残体。"], ["tomato_early", "tomato_spot"]),
    "Tomato___Late_blight": entry("适宜潮湿条件下可能迅速扩展，应优先开展田间复核。",
        "检查快速扩大的水渍状紫褐斑，并观察叶背、茎和果实。",
        ["联系植保人员尽快核查，并记录附近番茄、马铃薯发病情况。", "检查移栽苗健康状态，减少叶面淋水，清理自生寄主和病残体。"], ["tomato_late"], "高"),
    "Tomato___Leaf_Mold": entry("常与设施高湿环境相关，需结合叶背特征鉴别。",
        "检查叶面黄斑对应的叶背橄榄褐色霉层。",
        ["加强温室或棚内通风，降低凝露和长时间高湿。", "采用滴灌与合理株距，做好栽培支架、工具和季末残体管理。"], ["tomato_mold"]),
    "Tomato___Septoria_leaf_spot": entry("湿润条件及溅水可能促使传播，需与早疫病等区分。",
        "观察下部叶片小圆斑、深色边缘与斑内黑点。",
        ["合理支架和株距，使用覆盖物减少土壤溅水，尽量根部浇水。", "确诊后适度处理病叶，季末清理病残体，避免过度去叶。"], ["septoria"]),
    "Tomato___Spider_mites Two-spotted_spider_mite": entry("这是螨害类别，需先确认活螨，不能作为真菌病处理。",
        "用放大镜检查叶背活螨、细网和密集失绿小点。",
        ["减少干旱胁迫与扬尘，监测叶背螨量及天敌。", "保护捕食性天敌，避免不必要的广谱杀虫处理；需要防治时请植保人员制定方案。"], ["mites"]),
    "Tomato___Target_Spot": entry("可能与早疫病等混淆，冠层内部病情容易漏查。",
        "同时观察冠层内叶片、茎和果实，必要时送检区分病因。",
        ["使用健康苗，配合与非茄科作物轮作。", "管理杂草、自生寄主和病残体，调查隐蔽冠层病情后再确定针对性措施。"], ["target"]),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": entry("传播与粉虱及带病种苗有关，疑似感染应优先复核。",
        "卷叶黄化也可能有其他原因，应检测确认并调查粉虱。",
        ["使用健康苗和适合当地的抗性品种，并监测管理粉虱。", "对疑似病株送检；确诊后按当地指导控制病株与杂草等传播来源。"], ["tylcv"], "高"),
    "Tomato___Tomato_mosaic_virus": entry("可经接触、工具和种苗传播，应优先复核并防止机械传播。",
        "花叶与畸形症状并不能确认病毒种类，需检测鉴别。",
        ["注意手、工具和栽培材料的卫生，减少病健株之间交叉接触。", "使用健康种苗；确诊后按植保指导处理病株，不承诺药物治愈病毒感染。"], ["mosaic", "viruses"], "高"),
}


def validate_coverage(mapping):
    if set(mapping) != set(CHINESE_LABELS):
        raise ValueError("资料只适用于本项目冻结的39类")
    expected = {name for name in mapping if category_kind(name) == "病害 / 虫害类"}
    if set(DISEASE_CARE) != expected:
        raise ValueError("防治资料与冻结模型的病害/虫害类别覆盖不一致")
    for value in DISEASE_CARE.values():
        if value["risk_level"] not in ("中", "高") or not value["actions"]:
            raise ValueError("资料规则不完整")
        if not value["source_ids"] or any(key not in SOURCES for key in value["source_ids"]):
            raise ValueError("资料缺少有效来源")


def care_for_class(raw_name):
    if raw_name not in CHINESE_LABELS:
        raise ValueError(f"没有该类别的参考资料：{raw_name}")
    if raw_name in DISEASE_CARE:
        result = dict(DISEASE_CARE[raw_name])
    elif raw_name.endswith("___healthy"):
        result = entry("仅表示模型选中了健康类，不能排除未收录病害或田间其他风险。",
                       "如实物仍有异常，请补充其他叶片和整株观察并咨询植保人员。",
                       ["继续常规巡查并记录异常，不依据健康类预测自动施药。",
                        "保持合理栽培和卫生管理，出现持续异常时送诊。"], ["ipm"], "低")
    elif raw_name == "Background_without_leaves":
        result = entry("背景类别不提供作物病害风险判断。", "重新上传清晰单片叶片，核对作物种类。",
                       ["补拍叶片正反面及整株；本类别不能证明通用非叶片检测能力。"], [], "不适用")
    else:
        raise ValueError(f"没有该类别的参考资料：{raw_name}")
    result["class_name"] = raw_name
    result["reference_version"] = REFERENCE_VERSION
    result["sources"] = [{"title": SOURCES[key][0], "url": SOURCES[key][1], "checked_date": CHECKED_DATE}
                         for key in result["source_ids"]]
    return result


def build_prediction_report(result, image_info, final_record):
    """返回 Markdown 字符串供浏览器下载，不写入用户图片或报告文件。"""
    name = result["top3"][0]["class_name"]
    care = care_for_class(name)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    # 文件名只作为转义后的文本写进报告，不用作服务器路径。
    filename = str(image_info.get("filename", "未提供")).replace("`", "\\`").replace("\r", " ").replace("\n", " ")
    lines = ["# 植物叶片识别与防治参考报告", "", f"生成时间：{timestamp}",
             f"输入文件：`{filename}`", f"图片：{image_info.get('format', '未知')} / {image_info.get('width', '?')} × {image_info.get('height', '?')}",
             "", "## 分类结果（未经病原确诊）", "", "| 排名 | 中文名称 | 原始英文类别 | 模型概率 |",
             "| --- | --- | --- | ---: |"]
    for rank, item in enumerate(result["top3"], 1):
        lines.append(f"| {rank} | {display_name(item['class_name'])} | `{item['class_name']}` | {item['probability']:.2%} |")
    lines += ["", "预测概率是模型输出分数，不代表诊断可靠性。", "", "## 风险与严重程度", "",
              f"参考风险等级：**{care['risk_level']}（类别规则参考）**", care["reason"], "", RISK_SCOPE, "", SEVERITY_SCOPE,
              "", "## 复核与针对性防治参考", "", USE_SCOPE, "", f"复核要点：{care['signs_to_check']}"]
    lines += [f"- {text}" for text in care["actions"]]
    lines += ["", "资料来源（核对日期 " + CHECKED_DATE + "）："]
    lines += [f"- [{source['title']}]({source['url']})" for source in care["sources"]] or ["- 背景类别无病害处置建议。"]
    lines += ["", "## 模型与限制", "", f"模型文件：`{final_record['checkpoint']}`",
              f"模型 SHA256：`{result['checkpoint_sha256']}`", f"规则版本：`{REFERENCE_VERSION}`",
              "本报告没有更新模型或阈值。当前图片未保存到服务器；报告仅在点击下载后由浏览器保存。",
              "本划分已排除已知SHA256字节重复，但未排除增强同源关系，测试成绩可能偏乐观，不能直接代表独立叶片或真实拍摄场景的泛化能力。",
              "参考风险与防治模块尚未经过田间效果验证，不作为自动施药或检疫处置依据。", ""]
    return "\n".join(lines)
