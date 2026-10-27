"""Streamlit 中文单图展示。启动：项目 .venv 的 python -m streamlit run app.py。"""

import hashlib
import logging
from html import escape

from inference import (ALLOWED_FORMATS, FINAL_RECORD, SUPPORTED_FORMATS_TEXT,
                       InferenceError, decode_image, load_predictor,
                       model_cache_key, read_experiment_metrics, read_final_record)
from class_labels import category_kind, display_name
from smoke_utils import validate_mapping
from plant_care import (RISK_SCOPE, SEVERITY_SCOPE, USE_SCOPE, build_prediction_report,
                        care_for_class, validate_coverage)

import streamlit as st


@st.cache_resource(show_spinner=False)
def cached_predictor(cache_key):
    return load_predictor(cache_key[0])


def clear_prediction():
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_identity", None)


def supported_categories_panel():
    with st.expander("支持识别的类别（中英文，共39类）", expanded=False):
        try:
            mapping = read_final_record()["class_to_idx"]
            validate_mapping(mapping)
            rows = [{"中文名称": display_name(name), "英文类别名": name,
                     "类型": category_kind(name)}
                    for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
            counts = {kind: sum(row["类型"] == kind for row in rows)
                      for kind in ("病害 / 虫害类", "健康类", "背景类")}
            st.caption(f"共 {len(rows)} 类：病害 / 虫害 {counts['病害 / 虫害类']} 类、"
                       f"健康 {counts['健康类']} 类、背景 {counts['背景类']} 类；并非39种病害。")
            st.caption("鼠标拖选中文或英文，再按 Ctrl+C 即可复制；表格内可上下滚动。")
            # 使用真正的 HTML 文字表格，避免画布表格无法拖选文字。
            header = "".join(f"<th>{escape(column)}</th>" for column in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row.values())
                           + "</tr>" for row in rows)
            st.markdown("""<style>
            #category-reference-table {max-height:460px;overflow:auto;border:1px solid #d5dfd6;border-radius:8px;}
            #category-reference-table table {width:100%;border-collapse:collapse;font-size:14px;}
            #category-reference-table th {position:sticky;top:0;background:#edf2eb;z-index:1;text-align:left;}
            #category-reference-table th,#category-reference-table td {padding:10px 12px;border-bottom:1px solid #d5dfd6;
                user-select:text;-webkit-user-select:text;overflow-wrap:anywhere;}
            #category-reference-table td:first-child {min-width:180px;}
            #category-reference-table td:last-child {min-width:100px;}
            </style>""" + f'<div id="category-reference-table" tabindex="0" aria-label="可复制的中英文识别类别">'
                        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>",
                        unsafe_allow_html=True)
            st.caption("英文类别名与冻结模型保持一致；中文仅用于辅助理解。列表表示模型的分类范围，真实拍摄效果尚未验证。")
        except (InferenceError, KeyError, TypeError, ValueError) as exc:
            st.warning(f"无法读取支持类别：{exc}")


def experiment_panel():
    with st.expander("模型与实验结果", expanded=False):
        st.markdown("**ResNet18 迁移学习 · 39 类**　｜　第 3 轮冻结模型")
        try:
            metrics, path = read_experiment_metrics()
            left, right = st.columns(2)
            left.metric("固定测试集 Accuracy", f"{metrics['accuracy']:.2%}")
            right.metric("固定测试集 Macro F1", f"{metrics['macro_f1']:.2%}")
            st.caption(f"数据集实验成绩，来自已有真实结果文件；{metrics['correct']:,} / {metrics['total']:,} 张正确。")
            st.caption(f"指标来源：{path.name} · {path.parent.name}")
        except InferenceError as exc:
            st.warning(str(exc))
        st.write("已排除已知 SHA256 字节重复，增强同源关系尚未完全排除，成绩可能偏乐观。真实拍摄效果尚未验证。")
        st.caption("39 类包含健康叶片、病害 / 虫害和无叶片背景。背景类仅是本分类器的一类，不具备已验证的通用非叶片检测能力。")


def care_panel(result):
    with st.container(border=True):
        st.subheader("风险与防治参考")
        try:
            record = read_final_record()
            validate_coverage(record["class_to_idx"])
            care = care_for_class(result["top3"][0]["class_name"])
            st.markdown(f"**参考风险等级：{care['risk_level']}（类别规则参考）**")
            st.write(care["reason"])
            st.caption(RISK_SCOPE)
            st.caption(f"严重程度：{SEVERITY_SCOPE}")
            st.info("以下资料对应当前 Top1 预测；请先复核作物及病因，预测错误时建议不适用。")
            st.markdown(f"**复核要点：** {care['signs_to_check']}")
            for action in care["actions"]:
                st.markdown(f"- {action}")
            with st.expander("资料来源与适用范围", expanded=False):
                for source in care["sources"]:
                    st.markdown(f"[{source['title']}]({source['url']})（核对：{source['checked_date']}）")
                st.caption(USE_SCOPE)
            report = build_prediction_report(result, result.get("image_info", {}), record)
            st.download_button("下载本次识别与防治参考报告", data=report,
                               file_name="植物叶片识别与防治参考报告.md", mime="text/markdown",
                               on_click="ignore")
        except (KeyError, ValueError, InferenceError) as exc:
            st.error(f"参考资料暂不可用：{exc}")


def main():
    st.set_page_config(page_title="植物叶片病害识别系统", page_icon="🌿", layout="centered")
    st.markdown("""<style>
    .stApp {background: #f6f8f4;}
    .block-container {max-width: 1120px; padding-top: 2.3rem; padding-bottom: 2rem;}
    h1 {color: #193c2d; letter-spacing: -.04em;}
    [data-testid="stVerticalBlockBorderWrapper"] {background: white; border-radius: 16px;}
    .eyebrow {color: #477555; font-size: .82rem; font-weight: 650; letter-spacing: .14em;}
    [data-testid="stProgressBar"] > div > div {background-color: #397752;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">LEAF LAB / 植物视觉实验室</div>', unsafe_allow_html=True)
    st.title("植物叶片病害识别系统")
    st.write("上传一张叶片图片，查看39类识别结果、参考风险与防治资料。用于课程演示与学习。")
    st.caption("01 上传图片　→　02 开始识别　→　03 查看结果")
    supported_categories_panel()

    try:
        cache_key = model_cache_key(FINAL_RECORD)
        model_error = None
    except InferenceError as exc:
        cache_key, model_error = None, str(exc)
        clear_prediction()

    left, right = st.columns([1, 1.08], gap="large")
    image = None
    with left:
        with st.container(border=True):
            st.subheader("上传图片")
            uploaded = st.file_uploader("选择叶片图片", type=[suffix[1:] for suffix in ALLOWED_FORMATS],
                                        key="leaf_upload", on_change=clear_prediction,
                                        help=f"{SUPPORTED_FORMATS_TEXT}，最大 10 MiB、2000 万像素。图片只在内存中处理。")
            st.caption("支持 JPG、PNG、WebP、BMP、TIFF、静态 GIF、AVIF，以及 JPEG/JPE/JFIF 后缀。")
            identity = None
            if uploaded is not None:
                data = uploaded.getvalue()
                identity = (uploaded.name, hashlib.sha256(data).hexdigest(), cache_key)
                if st.session_state.get("prediction_identity") != identity:
                    clear_prediction()
                try:
                    image, info = decode_image(data, uploaded.name)
                    st.image(image, caption=info["filename"], width="stretch")
                    st.caption(f"{info['format']} · {info['width']} × {info['height']} 像素 · {info['bytes'] / 1024:.1f} KiB")
                except InferenceError as exc:
                    clear_prediction()
                    st.error(str(exc))
            else:
                clear_prediction()
                st.caption("图片默认只在本机内存中处理，不自动保存上传文件。")

    with right:
        with st.container(border=True):
            st.subheader("识别结果")
            if model_error:
                st.error(model_error)
            clicked = st.button("开始识别", type="primary", width="stretch",
                                disabled=image is None or model_error is not None)
            if clicked:
                clear_prediction()
                try:
                    with st.spinner("正在核对并加载模型、识别图片，请稍候……"):
                        predictor = cached_predictor(cache_key)
                        result = predictor.predict(image)
                        result["image_info"] = info
                    st.session_state["prediction"] = result
                    st.session_state["prediction_identity"] = identity
                except Exception as exc:
                    logging.exception("单图识别失败")
                    st.error(f"识别失败：{type(exc).__name__}: {exc}")
            result = st.session_state.get("prediction")
            if result and st.session_state.get("prediction_identity") == identity:
                best = result["top3"][0]
                st.caption(f"TOP 1 · {category_kind(best['class_name'])}")
                st.markdown(f"### {display_name(best['class_name'])}")
                st.code(best["class_name"], language=None)
                st.metric("预测概率", f"{best['probability']:.2%}")
                st.caption("预测概率是模型输出分数，不代表诊断可靠性。")
                st.divider()
                st.markdown("**Top 3 类别**")
                for rank, item in enumerate(result["top3"], 1):
                    st.write(f"{rank}. {display_name(item['class_name'])}　{item['probability']:.2%}")
                    st.progress(item["probability"])
                    st.caption(item["class_name"])
                st.caption(f"推理设备：{'GPU · CUDA' if result['device'].startswith('cuda') else 'CPU'} · 模型已冻结")
            elif not model_error:
                st.info("上传图片后，点击“开始识别”查看 Top 3 结果。")

    if image is not None:
        image.close()
    if result and st.session_state.get("prediction_identity") == identity:
        care_panel(result)
    st.divider()
    experiment_panel()
    st.caption("课程演示 · 单张图片识别 · 防治资料须经植保复核，不作为自动施药依据")


if __name__ == "__main__":
    main()
