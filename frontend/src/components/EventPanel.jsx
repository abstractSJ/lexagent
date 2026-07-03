import { memo, useEffect, useRef } from 'react';
import { Box, Button, Paper, Stack, Typography } from '@mui/material';
import EventCard from './EventCard.jsx';

/**
 * 右侧执行进度面板。
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
    <Paper
      elevation={0}
      component="aside"
      aria-labelledby="eventTitle"
      sx={{
        display: 'grid',
        gridTemplateRows: 'auto minmax(0, 1fr)',
        minHeight: 0,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 2,
        overflow: 'hidden',
        backgroundColor: 'rgba(255, 255, 255, 0.92)',
        boxShadow: '0 20px 50px rgba(20, 31, 54, 0.08)',
      }}
    >
      <Stack
        direction="row"
        spacing={2}
        sx={{
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 2.25,
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'rgba(248, 250, 252, 0.86)',
        }}
      >
        <Box>
          <Typography id="eventTitle" component="h2" variant="h6" sx={{ fontWeight: 900 }}>
            执行进度
          </Typography>
          <Typography color="text.secondary" sx={{ mt: 0.5, fontSize: 13 }}>
            这里展示法律咨询链路的实时事件。
          </Typography>
        </Box>
        <Button
          type="button"
          variant="text"
          size="small"
          onClick={onClear}
          sx={{ height: 30, minHeight: 30, minWidth: 'auto', px: 1.25, alignSelf: 'flex-start' }}
        >
          清空
        </Button>
      </Stack>

      <Stack
        ref={scrollRef}
        spacing={1.25}
        sx={{ minHeight: 0, overflowY: 'auto', p: 2 }}
        aria-live="polite"
        aria-label="执行事件列表"
      >
        {events.map((event) => (
          <EventCard key={event.id} event={event} />
        ))}
      </Stack>
    </Paper>
  );
});

export default EventPanel;
