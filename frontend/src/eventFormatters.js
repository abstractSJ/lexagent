/**
 * 前端事件展示格式化工具。
 *
 * 后端已经在 web_app/server.py 中完成敏感字段过滤；这里负责把安全事件转换成用户能读懂的
 * 标题、摘要和 Material UI 颜色语义。集中维护这些映射，是为了避免每个事件卡片组件都硬编码
 * 一套事件类型判断。
 */

const EVENT_TITLES = {
  legal_step: '执行步骤',
  legal_rag_query_started: '法条检索中',
  case_state_updated: '案件状态已更新',
  legal_missing_details_suggested: '可先补充的关键信息',
  legal_case_rag_done: '案情拆解与检索完成',
  legal_web_search_started: '公网案例与司法实践检索中',
  legal_web_search_done: '公网案例与司法实践检索完成',
  legal_reference_materials: '参考资料已整理',
  legal_risk_analyzed: '风险识别完成',
  legal_analysis_catalog_built: '案情目录已生成',
  legal_next_action_decided: '下一步动作已判断',
  tool_call: '工具调用',
  tool_result: '工具结果',
  message_done: '最终回答生成完成',
  done: '处理完成',
  waiting: '等待输入',
  error: '执行出错',
};

/**
 * 将事件类型转换为展示标题。
 *
 * @param {string} eventType 后端事件类型。
 * @returns {string} 中文展示名称。
 */
export function humanizeEventType(eventType) {
  return EVENT_TITLES[eventType] || eventType || '未知事件';
}

/**
 * 生成事件摘要。
 *
 * @param {string} eventType 后端事件类型。
 * @param {object} data 事件数据。
 * @returns {string} 卡片中的简短摘要。
 */
export function summarizeEvent(eventType, data = {}) {
  if (eventType === 'legal_step') {
    const status = data.status === 'start' ? '开始' : data.status || '进行中';
    return `${status}：${data.name || '未命名步骤'}`;
  }
  if (eventType === 'legal_rag_query_started') {
    return '正在检索本地法条，请稍候。';
  }
  if (eventType === 'case_state_updated') {
    return `案件状态已更新，版本：${data.version || 0}`;
  }
  if (eventType === 'legal_missing_details_suggested') {
    const questionCount = Array.isArray(data.questions) ? data.questions.length : 0;
    const evidenceCount = Array.isArray(data.evidence_gaps) ? data.evidence_gaps.length : 0;
    return `${data.message || '建议补充关键信息'}（问题 ${questionCount} 个，材料 ${evidenceCount} 项）`;
  }
  if (eventType === 'legal_case_rag_done') {
    return '本地法条检索完成。';
  }
  if (eventType === 'legal_web_search_started') {
    return '正在检索公网案例、裁判规则和司法实践资料。';
  }
  if (eventType === 'legal_web_search_done') {
    return `公网资料检索完成，找到 ${data.result_count || 0} 条，提示 ${data.warning_count || 0} 项。`;
  }
  if (eventType === 'legal_reference_materials') {
    const lawCount = Array.isArray(data.laws) ? data.laws.length : 0;
    const webCount = Array.isArray(data.web) ? data.web.length : 0;
    return `已整理 ${lawCount} 条法条资料、${webCount} 条案例/实务资料，可在右侧参考资料栏查看。`;
  }
  if (eventType === 'legal_risk_analyzed') {
    return `识别风险 ${data.risk_count || 0} 项。`;
  }
  if (eventType === 'legal_analysis_catalog_built') {
    return `追问 ${data.follow_up_question_count || 0} 个，法律概念 ${data.legal_concept_count || 0} 个。`;
  }
  if (eventType === 'legal_next_action_decided') {
    const correction = data.should_correct_previous_answer ? '，需修正前序判断' : '';
    return `${data.action || '未指定动作'}${correction}。`;
  }
  if (eventType === 'tool_call') {
    return '正在处理必要工具。';
  }
  if (eventType === 'tool_result') {
    return '工具处理完成。';
  }
  if (eventType === 'message_done') {
    return '最终回答文本已生成。';
  }
  if (eventType === 'error') {
    return data.error || '执行过程中发生错误。';
  }
  if (eventType === 'done') {
    return '本轮请求已经结束。';
  }
  if (eventType === 'waiting') {
    return data.message || '发送问题后，会看到案件状态更新、检索状态和处理进度。';
  }
  return stringifyData(data);
}

/**
 * 把事件类型映射成 Material UI 颜色。
 *
 * @param {string} eventType 后端事件类型。
 * @returns {'default'|'primary'|'secondary'|'error'|'info'|'success'} MUI 颜色名。
 */
export function colorForEvent(eventType) {
  if (eventType === 'error') {
    return 'error';
  }
  if (eventType === 'done' || eventType === 'message_done') {
    return 'success';
  }
  if (
    eventType === 'legal_rag_query_started' ||
    eventType === 'legal_web_search_started' ||
    eventType === 'legal_web_search_done' ||
    eventType === 'legal_reference_materials' ||
    eventType === 'legal_missing_details_suggested' ||
    eventType === 'tool_call' ||
    eventType === 'tool_result'
  ) {
    return 'info';
  }
  if (eventType === 'legal_step') {
    return 'secondary';
  }
  return 'default';
}

/**
 * 返回事件卡片的柔和背景色。
 *
 * @param {string} eventType 后端事件类型。
 * @returns {{borderColor: string, backgroundColor: string, titleColor: string}} 卡片色彩配置。
 */
export function toneForEvent(eventType) {
  if (eventType === 'error') {
    return { borderColor: '#fecdd3', backgroundColor: '#fff1f3', titleColor: '#b42318' };
  }
  if (eventType === 'done' || eventType === 'message_done') {
    return { borderColor: '#bbf7d0', backgroundColor: '#ecfdf3', titleColor: '#15803d' };
  }
  if (
    eventType === 'legal_rag_query_started' ||
    eventType === 'legal_web_search_started' ||
    eventType === 'legal_web_search_done' ||
    eventType === 'legal_reference_materials' ||
    eventType === 'legal_missing_details_suggested' ||
    eventType === 'tool_call' ||
    eventType === 'tool_result'
  ) {
    return { borderColor: '#bae6fd', backgroundColor: '#eef6ff', titleColor: '#0369a1' };
  }
  if (eventType === 'legal_step') {
    return { borderColor: '#ddd6fe', backgroundColor: '#f4f0ff', titleColor: '#7c3aed' };
  }
  return { borderColor: '#d9e0ea', backgroundColor: '#f8fafc', titleColor: '#18202f' };
}

/**
 * 把顶部状态类型映射成 MUI Chip 颜色。
 *
 * @param {'ok'|'pending'|'error'} kind 状态类型。
 * @returns {'success'|'info'|'error'|'default'} MUI Chip 颜色。
 */
export function colorForStatus(kind) {
  if (kind === 'ok') {
    return 'success';
  }
  if (kind === 'error') {
    return 'error';
  }
  if (kind === 'pending') {
    return 'info';
  }
  return 'default';
}

/**
 * 构造补充内容在聊天区展示的文本。
 *
 * @param {object} payload 补充请求体。
 * @returns {string} 用户消息展示文本。
 */
export function buildSupplementDisplayText(payload) {
  const lines = ['我补充以下关键信息：'];
  for (const [question, answer] of Object.entries(payload.supplement_answers || {})) {
    lines.push(`- ${question}：${answer}`);
  }
  if (payload.selected_evidence_gaps?.length) {
    lines.push('已确认/准备的证据材料：');
    payload.selected_evidence_gaps.forEach((item) => lines.push(`- ${item}`));
  }
  if (payload.free_text) {
    lines.push(payload.free_text);
  }
  return lines.join('\n');
}

/**
 * 安全序列化兜底事件数据。
 *
 * @param {object} data 事件数据。
 * @returns {string} 可展示文本。
 */
function stringifyData(data) {
  try {
    return JSON.stringify(data);
  } catch {
    return '收到无法序列化的事件数据。';
  }
}
