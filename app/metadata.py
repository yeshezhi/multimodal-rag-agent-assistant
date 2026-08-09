from typing import Any


DEFAULT_METADATA = {
    "department": "未分类",
    "document_type": "其他",
    "classification": "内部",
    "effective_date": "未标注",
    "tags": [],
}


DEMO_METADATA: dict[str, dict[str, Any]] = {
    "员工入职与远程办公指南.md": {"department": "人力资源部", "document_type": "员工制度", "classification": "内部", "effective_date": "2026-01-01", "tags": ["入职", "远程办公", "考勤"]},
    "服务器GPU故障处理手册.md": {"department": "基础设施部", "document_type": "运维手册", "classification": "内部", "effective_date": "2026-02-01", "tags": ["GPU", "故障", "P1"]},
    "项目交付与验收规范.md": {"department": "交付管理部", "document_type": "项目规范", "classification": "内部", "effective_date": "2026-01-15", "tags": ["交付", "验收", "缺陷"]},
    "信息安全与账号权限管理制度.md": {"department": "信息安全部", "document_type": "安全制度", "classification": "受限", "effective_date": "2026-03-01", "tags": ["账号", "MFA", "权限"]},
    "IT服务台与终端设备支持指南.md": {"department": "信息技术部", "document_type": "服务指南", "classification": "内部", "effective_date": "2026-02-15", "tags": ["IT服务台", "终端", "P1"]},
    "模型训练平台资源管理规范.md": {"department": "AI平台部", "document_type": "平台规范", "classification": "内部", "effective_date": "2026-03-10", "tags": ["训练", "GPU", "资源"]},
    "客户数据与项目文档管理规范.md": {"department": "数据治理部", "document_type": "数据制度", "classification": "受限", "effective_date": "2026-03-20", "tags": ["客户数据", "文档", "保留"]},
}


def normalize_metadata(metadata: dict[str, Any] | None, source_name: str = "") -> dict[str, Any]:
    result = {**DEFAULT_METADATA, **DEMO_METADATA.get(source_name, {}), **(metadata or {})}
    tags = result.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.replace("，", ",").split(",") if tag.strip()]
    result["tags"] = list(dict.fromkeys(str(tag).strip() for tag in tags if str(tag).strip()))
    for key in ("department", "document_type", "classification", "effective_date"):
        result[key] = str(result.get(key) or DEFAULT_METADATA[key]).strip()
    return result


def matches_metadata(
    metadata: dict[str, Any], department: str | None, document_type: str | None, tag: str | None
) -> bool:
    if department and metadata["department"] != department:
        return False
    if document_type and metadata["document_type"] != document_type:
        return False
    if tag:
        normalized_tag = tag.strip().lower()
        if normalized_tag and not any(normalized_tag in item.lower() for item in metadata["tags"]):
            return False
    return True
