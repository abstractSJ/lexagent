import { memo } from 'react';
import { Box, Typography } from '@mui/material';
import { humanizeEventType, summarizeEvent, toneForEvent } from '../eventFormatters.js';
import {
  ActivityIcon,
  AlertTriangleIcon,
  BoltIcon,
  BookOpenIcon,
  ChatIcon,
  CheckCircleIcon,
  DatabaseIcon,
  FileTextIcon,
  GlobeIcon,
  SparklesIcon,
} from '../icons.jsx';

/**
 * 事件类型到时间线节点图标的映射。
 *
 * 图标语义和 toneForEvent 的色调配套：错误用警告三角、本地法条检索用数据库、公网检索用地球、
 * 分析结论类用星光。集中成映射表而不是 if 链，是为了新增事件类型时只加一行，不用改渲染逻辑。
 */
const EVENT_ICONS = {
  // 错误：警告三角，配合红色调一眼可辨。
  error: AlertTriangleIcon,
  // 自修复本质是"链路波动后自动恢复"，用闪电强调动作发生，而不是失败。
  legal_selfheal: BoltIcon,
  // 完成类：对勾圆圈。
  done: CheckCircleIcon,
  message_done: CheckCircleIcon,
  // 本地法条 RAG 检索：数据库。
  legal_rag_query_started: DatabaseIcon,
  legal_case_rag_done: DatabaseIcon,
  // 公网案例/司法实践检索：地球。
  legal_web_search_started: GlobeIcon,
  legal_web_search_done: GlobeIcon,
  // 参考资料整理：打开的书，与右侧资料栏的身份图标一致。
  legal_reference_materials: BookOpenIcon,
  // 历史记忆唤起：星光。
  legal_memory_recalled: SparklesIcon,
  // 案件状态/补充信息类：文档。
  legal_missing_details_suggested: FileTextIcon,
  case_state_updated: FileTextIcon,
  // 轮级指标：心电波形。
  legal_turn_metrics: ActivityIcon,
  // 等待输入：对话气泡。
  waiting: ChatIcon,
  // 工具调用过程：闪电。
  tool_call: BoltIcon,
  tool_result: BoltIcon,
  // 综合分析产出（风险/目录/下一步动作）：星光。
  legal_risk_analyzed: SparklesIcon,
  legal_analysis_catalog_built: SparklesIcon,
  legal_next_action_decided: SparklesIcon,
  // 通用执行步骤：心电波形。
  legal_step: ActivityIcon,
};

/**
 * 根据事件类型取时间线节点图标组件。
 *
 * @param {string} eventType 后端事件类型。
 * @returns {(props: object) => JSX.Element} 内联 SVG 图标组件；未知类型回退到心电波形。
 */
function iconForEvent(eventType) {
  return EVENT_ICONS[eventType] || ActivityIcon;
}

/**
 * 单个执行事件的时间线节点行。
 *
 * 视觉结构：左列彩色节点圆压在时间线竖线上（竖线由 EventPanel 的时间线容器绘制），
 * 右列是标题加摘要。相比旧的方块卡片，时间线形态更贴合"链路按步骤推进"的心智模型，
 * 也去掉了与标题重复的类型 Chip。
 *
 * 使用 memo 的原因是事件列表只增不改；新事件到达时旧节点的 props 引用不变，可整体跳过重渲染。
 *
 * @param {object} props 组件参数。
 * @param {object} props.event 后端规范化后的事件或前端生成的 done/error 事件。
 * @returns {JSX.Element} 时间线节点行。
 */
const EventCard = memo(function EventCard({ event }) {
  const eventType = event.event_type || event.type || 'unknown';
  const data = event.data || {};
  const tone = toneForEvent(eventType);
  const title = event.title || humanizeEventType(eventType);
  const summary = summarizeEvent(eventType, data);
  const NodeIcon = iconForEvent(eventType);

  return (
    <Box
      sx={{
        position: 'relative',
        display: 'grid',
        gridTemplateColumns: '32px 1fr',
        columnGap: 1.25,
        py: 1,
        // 事件列表在纵向滚动容器里；节点行必须禁止收缩，否则事件多时会被压成横线。
        flexShrink: 0,
      }}
    >
      <Box
        aria-hidden="true"
        sx={{
          width: 28,
          height: 28,
          // 28px 圆在 32px 列内水平居中后，圆心正好落在时间线竖线（left 15px、宽 2px）的中心上。
          justifySelf: 'center',
          zIndex: 1,
          display: 'grid',
          placeItems: 'center',
          borderRadius: '50%',
          backgroundColor: tone.iconBg,
          color: tone.fg,
          // 白色描边把节点底下的竖线盖住，形成"珠子串在线上"的效果。
          border: '2px solid #fff',
          boxShadow: '0 2px 8px -2px rgba(21, 32, 54, 0.18)',
        }}
      >
        <NodeIcon sx={{ fontSize: 14 }} />
      </Box>
      <Box sx={{ minWidth: 0 }}>
        <Typography sx={{ fontSize: 13.5, fontWeight: 700, color: tone.titleColor, lineHeight: 1.4 }}>
          {title}
        </Typography>
        <Typography
          sx={{ mt: 0.25, fontSize: 12.5, lineHeight: 1.55, color: 'text.secondary', wordBreak: 'break-word' }}
        >
          {summary}
        </Typography>
      </Box>
    </Box>
  );
});

export default EventCard;
