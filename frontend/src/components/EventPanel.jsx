import { memo, useEffect, useRef } from 'react';
import { Box, Button, Stack, Typography } from '@mui/material';
import EventCard from './EventCard.jsx';
import PanelShell from './PanelShell.jsx';
import { ActivityIcon } from '../icons.jsx';

/**
 * 左侧执行进度面板。
 *
 * 事件以时间线形式呈现：竖线由本组件的时间线容器统一绘制，EventCard 的节点圆压在线上。
 * 竖线画在容器伪元素而不是每个节点内部，是为了避免节点行间距变化时出现断线。
 *
 * 使用 memo 的原因是最终回答流式期间 events 引用不变，本面板可整体跳过高频增量渲染。
 *
 * @param {object} props 组件参数。
 * @param {object[]} props.events 事件列表。
 * @param {() => void} props.onClear 清空事件。
 * @returns {JSX.Element} 执行进度面板。
 */
const EventPanel = memo(function EventPanel({ events, onClear }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    // 进度区始终跟随最新事件。这里不做“用户上翻则停住”的判断，原因是事件卡片是概括性状态，
    // 用户主要关心当前进行到哪一步，历史事件可以在本轮结束后再回看。
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <PanelShell
      component="aside"
      titleId="eventTitle"
      icon={<ActivityIcon />}
      accent={{ bg: '#f0edfd', fg: '#5b52d6' }}
      title="执行进度"
      subtitle="这里展示法律咨询链路的实时事件。"
      action={
        <Button
          type="button"
          variant="text"
          size="small"
          onClick={onClear}
          sx={{ flexShrink: 0, minWidth: 'auto', px: 1.25 }}
        >
          清空
        </Button>
      }
    >
      <Box
        ref={scrollRef}
        sx={{ minHeight: 0, overflowY: 'auto', px: 2, py: 1.75 }}
        aria-live="polite"
        aria-label="执行事件列表"
      >
        {events.length === 0 ? (
          <Stack spacing={1} sx={{ py: 5, alignItems: 'center' }}>
            <ActivityIcon sx={{ fontSize: 28, color: '#c3cede' }} />
            <Typography sx={{ fontSize: 13, color: 'text.secondary' }}>暂无执行事件</Typography>
          </Stack>
        ) : (
          <Box
            sx={{
              position: 'relative',
              // 时间线竖线：贯穿全部节点，位置对准 EventCard 左列 32px 内居中的 28px 节点圆圆心。
              '&::before': {
                content: '""',
                position: 'absolute',
                left: '15px',
                top: '10px',
                bottom: '10px',
                width: '2px',
                backgroundColor: '#e6ebf6',
                borderRadius: 1,
              },
            }}
          >
            {events.map((event) => (
              <EventCard key={event.id} event={event} />
            ))}
          </Box>
        )}
      </Box>
    </PanelShell>
  );
});

export default EventPanel;
