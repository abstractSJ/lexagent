/**
 * 前端事件展示格式化工具。
 *
 * 后端已经在 web_app/server.py 中完成敏感字段过滤；这里负责把安全事件转换成用户能读懂的
 * 标题、摘要和 Material UI 颜色语义。集中维护这些映射，是为了避免每个事件卡片组件都硬编码
 * 一套事件类型判断。
 */

const EVENT_TITLES = {
  legal_step: '执行步骤',
  legal_selfheal: '链路自修复',
  legal_memory_recalled: '历史咨询记忆已唤起',
  legal_turn_metrics: '本轮执行指标',
  legal_rag_query_started: '法条检索中',
  case_state_updated: '案件状态已更新',
  legal_missing_details_suggested: '可先补充的关键信息',
  legal_supplement_skipped: '已按现有信息继续分析',
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
    // 前端自造的步骤事件（载入历史会话、补充暂存等）只带 message，没有 name/status；
    // 直接展示 message，避免渲染成"进行中：未命名步骤"这类占位文案。
    if (!data.name && data.message) {
      return data.message;
    }
    const status = data.status === 'start' ? '开始' : data.status || '进行中';
    return `${status}：${data.name || '未命名步骤'}`;
  }
  if (eventType === 'legal_selfheal') {
    const stage = data.stage || '内部环节';
    return data.action === 'retried'
      ? `${stage}出现波动，已自动重试成功。`
      : `${stage}暂时失败，已降级继续，本轮结论可能不完整。`;
  }
  if (eventType === 'legal_memory_recalled') {
    const memories = Array.isArray(data.memories) ? data.memories : [];
    const titles = memories
      .map((item) => item?.title)
      .filter(Boolean)
      .slice(0, 3)
      .join('、');
    const count = data.count || memories.length;
    return titles ? `结合 ${count} 条历史咨询：${titles}` : `结合 ${count} 条历史咨询记忆。`;
  }
  if (eventType === 'legal_turn_metrics') {
    const usage = data.llm_usage || {};
    const seconds = ((data.total_duration_ms || 0) / 1000).toFixed(1);
    const base = `总耗时 ${seconds} 秒，LLM 调用 ${usage.calls || 0} 次、${usage.total_tokens || 0} tokens。`;
    const selfhealCount = data.selfheal_count || 0;
    return selfhealCount > 0 ? `${base}本轮自修复 ${selfhealCount} 次。` : base;
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
  if (eventType === 'legal_supplement_skipped') {
    return '已按你的要求跳过补充，基于现有信息继续检索和分析。';
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
  if (eventType === 'legal_selfheal') {
    return 'info';
  }
  if (eventType === 'done' || eventType === 'message_done') {
    return 'success';
  }
  if (
    eventType === 'legal_rag_query_started' ||
    eventType === 'legal_memory_recalled' ||
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
 * 返回事件的展示色调。
 *
 * 时间线节点视觉主要消费 iconBg（节点底色）、fg（节点图标色）和 titleColor（标题色）；
 * borderColor 和 backgroundColor 继续保留并取相近的新色值，是为了兼容仍按旧字段取色的
 * 引用方，避免升级时间线样式时波及其他展示位。
 *
 * @param {string} eventType 后端事件类型。
 * @returns {{borderColor: string, backgroundColor: string, titleColor: string, iconBg: string, fg: string}} 事件色彩配置。
 */
export function toneForEvent(eventType) {
  if (eventType === 'error') {
    return { borderColor: '#f5c2c0', backgroundColor: '#fdf2f2', titleColor: '#b42318', iconBg: '#fbe3e1', fg: '#c0392b' };
  }
  if (eventType === 'legal_selfheal') {
    // 自修复用琥珀色区分于普通 info：提醒用户本轮有环节波动，但不是失败红色。
    return { borderColor: '#f0dcae', backgroundColor: '#fdf7e7', titleColor: '#9a6410', iconBg: '#f9eccd', fg: '#b7791f' };
  }
  if (eventType === 'done' || eventType === 'message_done') {
    return { borderColor: '#c2e5d2', backgroundColor: '#effaf3', titleColor: '#0f8a5f', iconBg: '#d9f2e4', fg: '#0f8a5f' };
  }
  if (
    eventType === 'legal_rag_query_started' ||
    eventType === 'legal_memory_recalled' ||
    eventType === 'legal_web_search_started' ||
    eventType === 'legal_web_search_done' ||
    eventType === 'legal_reference_materials' ||
    eventType === 'legal_missing_details_suggested' ||
    eventType === 'tool_call' ||
    eventType === 'tool_result'
  ) {
    return { borderColor: '#c9dbf3', backgroundColor: '#f0f6ff', titleColor: '#1e5fae', iconBg: '#dce9fb', fg: '#2270c8' };
  }
  if (eventType === 'legal_step') {
    return { borderColor: '#d9d6f6', backgroundColor: '#f5f4ff', titleColor: '#5b52d6', iconBg: '#e7e4fa', fg: '#5b52d6' };
  }
  return { borderColor: '#e3e8f3', backgroundColor: '#f8fafd', titleColor: '#3c4a66', iconBg: '#edf1f8', fg: '#5c6a84' };
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
  if (payload.skip_supplement) {
    // 跳过补充时聊天区展示一句明确声明，和后端合成给模型的输入语义保持一致。
    return '我暂时无法补充这些信息，请基于现有信息继续分析。';
  }
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
