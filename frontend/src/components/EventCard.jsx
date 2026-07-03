import { memo } from 'react';
import { Card, CardContent, Chip, Stack, Typography } from '@mui/material';
import { colorForEvent, humanizeEventType, summarizeEvent, toneForEvent } from '../eventFormatters.js';

/**
 * 单个执行事件卡片。
 *
 * 使用 memo 的原因是事件列表只增不改；新事件到达时旧卡片的 props 引用不变，可整体跳过重渲染。
 *
 * @param {object} props 组件参数。
 * @param {object} props.event 后端规范化后的事件或前端生成的 done/error 事件。
 * @returns {JSX.Element} 事件卡片。
 */
const EventCard = memo(function EventCard({ event }) {
  const eventType = event.event_type || event.type || 'unknown';
  const data = event.data || {};
  const tone = toneForEvent(eventType);
  const title = event.title || humanizeEventType(eventType);
  const summary = summarizeEvent(eventType, data);

  return (
    <Card
      variant="outlined"
      sx={{
        borderColor: tone.borderColor,
        backgroundColor: tone.backgroundColor,
        borderRadius: 1.5,
        // 事件列表本身是纵向 flex 滚动容器；卡片必须禁止收缩，否则事件多时会被压成横线。
        flexShrink: 0,
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Stack spacing={0.75}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography sx={{ color: tone.titleColor, fontWeight: 900, lineHeight: 1.35 }}>{title}</Typography>
            <Chip size="small" label={humanizeEventType(eventType)} color={colorForEvent(eventType)} variant="outlined" />
          </Stack>
          <Typography sx={{ fontSize: 13, lineHeight: 1.55, wordBreak: 'break-word' }}>{summary}</Typography>
        </Stack>
      </CardContent>
    </Card>
  );
});

export default EventCard;
