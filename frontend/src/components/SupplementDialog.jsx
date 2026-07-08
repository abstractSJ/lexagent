import { memo, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  FormGroup,
  IconButton,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { AlertTriangleIcon, CloseIcon, FileTextIcon } from '../icons.jsx';

// 问题/证据卡片的共用外观：白底圆角 + 细边框，hover 时边框轻微加深提示可交互区域。
// 抽成常量保证两个分组里的卡片始终同款，微调时不会只改到其中一处。
const ITEM_CARD_SX = {
  backgroundColor: '#fff',
  borderRadius: '12px',
  border: '1px solid #e3e9f4',
  transition: 'border-color 0.18s ease',
  '&:hover': { borderColor: '#c8d4ee' },
};

/**
 * 补充信息弹窗。
 *
 * 同一个弹窗同时服务两类场景：
 * 1. blocking pause：后端要求先补充，主输入保持禁用；
 * 2. optional suggestion：后端只是建议补充，用户可以点击按钮主动补充，也可以继续普通追问。
 *
 * 保留逐项 checkbox 的原因是用户不一定能一次回答所有问题；被勾选的问题会随请求发送给后端，
 * 有答案的项会进入 supplement_answers，没有答案但被勾选的项会进入 selected_questions。
 *
 * 使用 memo 的原因是弹窗常在最终回答仍在流式生成时被打开编辑；所有 props 在流式期间
 * 均保持引用稳定，memo 让高频 answer_delta 渲染完全跳过弹窗子树，保证输入不被拖慢。
 *
 * @param {object} props 组件参数。
 * @param {object|null} props.supplement 补充问题数据。
 * @param {boolean} props.open 是否打开弹窗。
 * @param {boolean} props.blocking 是否为阻塞性补充。
 * @param {boolean} props.disabled 是否禁用提交。
 * @param {() => void} props.onClose 关闭弹窗。
 * @param {(payload: object) => Promise<boolean|void>|boolean|void} props.onContinue 提交补充信息。
 * @returns {JSX.Element|null} 补充信息弹窗。
 */
const SupplementDialog = memo(function SupplementDialog({ supplement, open, blocking, disabled, onClose, onContinue }) {
  const questions = useMemo(() => (Array.isArray(supplement?.questions) ? supplement.questions : []), [supplement]);
  const evidenceGaps = useMemo(
    () => (Array.isArray(supplement?.evidence_gaps) ? supplement.evidence_gaps : []),
    [supplement],
  );
  const [selectedQuestions, setSelectedQuestions] = useState({});
  const [answers, setAnswers] = useState({});
  const [selectedEvidenceGaps, setSelectedEvidenceGaps] = useState({});
  const [freeText, setFreeText] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!supplement) {
      return;
    }

    const draft = supplement.draft_payload || {};
    const draftAnswers = draft.supplement_answers && typeof draft.supplement_answers === 'object' ? draft.supplement_answers : {};
    const draftQuestions = Array.isArray(draft.selected_questions) ? new Set(draft.selected_questions) : null;
    const draftEvidence = Array.isArray(draft.selected_evidence_gaps) ? new Set(draft.selected_evidence_gaps) : null;

    // 新一组问题默认勾选全部项；如果自动提交失败后恢复弹窗，则用 draft_payload 还原用户刚才填写的草稿，
    // 避免网络错误或后端忙碌时丢掉补充内容。
    setSelectedQuestions(Object.fromEntries(questions.map((item) => [item, draftQuestions ? draftQuestions.has(item) : true])));
    setAnswers(Object.fromEntries(questions.map((item) => [item, draftAnswers[item] || '']).filter(([, value]) => value)));
    setSelectedEvidenceGaps(Object.fromEntries(evidenceGaps.map((item) => [item, draftEvidence ? draftEvidence.has(item) : true])));
    setFreeText(typeof draft.free_text === 'string' ? draft.free_text : '');
    setError('');
  }, [supplement, questions, evidenceGaps]);

  if (!supplement) {
    return null;
  }

  const handleSubmit = async () => {
    const supplementAnswers = {};
    const selectedQuestionList = [];
    for (const question of questions) {
      if (!selectedQuestions[question]) {
        continue;
      }
      selectedQuestionList.push(question);
      const answer = (answers[question] || '').trim();
      if (answer) {
        supplementAnswers[question] = answer;
      }
    }

    const selectedEvidenceList = evidenceGaps.filter((item) => selectedEvidenceGaps[item]);
    const trimmedFreeText = freeText.trim();
    const hasAnswers = Object.keys(supplementAnswers).length > 0;
    const hasEvidence = selectedEvidenceList.length > 0;
    const hasFreeText = Boolean(trimmedFreeText);

    if (!hasAnswers && !hasEvidence && !hasFreeText) {
      setError('请先填写补充内容，或勾选可以补充/说明的证据材料。');
      return;
    }

    setError('');
    // 弹窗的开关完全交给上层管理：提交被接受时上层会同步关闭弹窗，失败时再恢复重开。
    // 这里不能在 await 之后自行调用 onClose——onContinue 可能要等整轮流式响应结束才
    // resolve，那时如果本轮又产生了新的补充建议弹窗，迟到的 onClose 会把它误关掉。
    await onContinue({
      message: '我补充以下关键信息：',
      supplement_answers: supplementAnswers,
      selected_questions: selectedQuestionList,
      selected_evidence_gaps: selectedEvidenceList,
      free_text: trimmedFreeText,
    });
  };

  const handleSkip = async () => {
    // 用户确实没有更多信息时不能把流程卡死：跳过表单校验，直接请求后端基于现有信息
    // 继续完整分析链路。展示文案和排队/失败恢复逻辑复用普通补充提交的同一条路径。
    setError('');
    await onContinue({
      message: '',
      skip_supplement: true,
    });
  };

  return (
    <Dialog open={open} onClose={disabled ? undefined : onClose} fullWidth maxWidth="md" aria-labelledby="supplementDialogTitle">
      <DialogTitle
        id="supplementDialogTitle"
        sx={{ display: 'flex', alignItems: 'center', gap: 1.5, pb: 1 }}
      >
        {/* 图标块用琥珀/品牌蓝区分“必须补充”与“建议补充”，与聊天区补充入口卡片的色彩语义一致。 */}
        <Box
          aria-hidden="true"
          sx={{
            width: 34,
            height: 34,
            flexShrink: 0,
            display: 'grid',
            placeItems: 'center',
            borderRadius: '10px',
            backgroundColor: blocking ? '#fbead0' : '#e8edfb',
            color: blocking ? '#b45309' : '#2b4ecb',
          }}
        >
          {blocking ? <AlertTriangleIcon sx={{ fontSize: 17 }} /> : <FileTextIcon sx={{ fontSize: 17 }} />}
        </Box>
        <Box component="span" sx={{ flex: 1, minWidth: 0, fontWeight: 800, fontSize: 17 }}>
          {blocking ? '请先补充关键信息' : '补充关键信息'}
        </Box>
        <IconButton
          type="button"
          aria-label="关闭补充信息弹窗"
          disabled={disabled}
          onClick={onClose}
          sx={{ color: 'text.secondary' }}
        >
          <CloseIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers sx={{ backgroundColor: '#f6f8fd' }}>
        <Stack spacing={1.5}>
          <Alert severity={blocking ? 'warning' : 'info'} sx={{ borderRadius: '12px' }}>
            {supplement.reason || supplement.message || '这些信息会帮助后续法律检索和分析更准确。'}
          </Alert>

          {questions.length > 0 && (
            <SupplementSection title="需要确认的问题">
              {questions.map((question, index) => (
                <Paper key={question} elevation={0} sx={{ p: 1.5, ...ITEM_CARD_SX }}>
                  <Stack spacing={1}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={Boolean(selectedQuestions[question])}
                          disabled={disabled}
                          onChange={(event) =>
                            setSelectedQuestions((current) => ({ ...current, [question]: event.target.checked }))
                          }
                        />
                      }
                      label={
                        // 题号做成圆形徽章放进 label：与问题文字一起构成 checkbox 的
                        // 可点击区域，长问题换行时序号也不会挤进正文。
                        // 顶部留 pt 是为了补偿 checkbox 自带的内边距，让首行与勾选框光学对齐。
                        <Stack direction="row" spacing={1} sx={{ alignItems: 'flex-start', pt: 1 }}>
                          <Box
                            sx={{
                              width: 22,
                              height: 22,
                              flexShrink: 0,
                              display: 'grid',
                              placeItems: 'center',
                              borderRadius: '50%',
                              backgroundColor: '#e8edfb',
                              color: '#2b4ecb',
                              fontSize: 12,
                              fontWeight: 700,
                            }}
                          >
                            {index + 1}
                          </Box>
                          <Typography sx={{ fontSize: 14, lineHeight: 1.5 }}>{question}</Typography>
                        </Stack>
                      }
                      sx={{ alignItems: 'flex-start', m: 0 }}
                    />
                    <TextField
                      value={answers[question] || ''}
                      onChange={(event) => setAnswers((current) => ({ ...current, [question]: event.target.value }))}
                      disabled={disabled || !selectedQuestions[question]}
                      multiline
                      minRows={2}
                      fullWidth
                      placeholder="在这里回答这个问题。"
                    />
                  </Stack>
                </Paper>
              ))}
            </SupplementSection>
          )}

          {evidenceGaps.length > 0 && (
            <SupplementSection title="可补充或说明的证据材料">
              <FormGroup sx={{ gap: 1 }}>
                {evidenceGaps.map((item) => (
                  <Paper key={item} elevation={0} sx={{ px: 1.5, py: 0.75, ...ITEM_CARD_SX }}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={Boolean(selectedEvidenceGaps[item])}
                          disabled={disabled}
                          onChange={(event) =>
                            setSelectedEvidenceGaps((current) => ({ ...current, [item]: event.target.checked }))
                          }
                        />
                      }
                      label={<Typography sx={{ fontSize: 14, lineHeight: 1.5 }}>{item}</Typography>}
                      sx={{ alignItems: 'flex-start', m: 0 }}
                    />
                  </Paper>
                ))}
              </FormGroup>
            </SupplementSection>
          )}

          <TextField
            label="其他补充说明"
            value={freeText}
            onChange={(event) => setFreeText(event.target.value)}
            disabled={disabled}
            multiline
            minRows={3}
            placeholder="例如：时间、金额、地点、对方说法、已有证据、是否报警/仲裁/起诉等。"
          />

          {error && <Alert severity="error" sx={{ borderRadius: '12px' }}>{error}</Alert>}
        </Stack>
      </DialogContent>
      <DialogActions
        sx={{ px: 3, py: 2, borderTop: '1px solid', borderColor: 'divider', backgroundColor: '#fbfcff' }}
      >
        {blocking && (
          // 阻塞暂停时给一条明确出路：确实无法补充也能让分析继续，避免用户被卡死在补充环节。
          // 非阻塞建议本来就不拦流程，“先不补充”关闭弹窗即可，不需要这个按钮。
          <Button type="button" variant="text" color="warning" disabled={disabled} onClick={() => void handleSkip()} sx={{ mr: 'auto' }}>
            无法补充，直接分析
          </Button>
        )}
        <Button type="button" variant="text" disabled={disabled} onClick={onClose}>
          先不补充
        </Button>
        <Button type="button" variant="contained" disabled={disabled} onClick={() => void handleSubmit()}>
          提交补充并继续
        </Button>
      </DialogActions>
    </Dialog>
  );
});

export default SupplementDialog;

/**
 * 弹窗内的补充分组。
 *
 * @param {object} props 组件参数。
 * @param {string} props.title 分组标题。
 * @param {React.ReactNode} props.children 分组内容。
 * @returns {JSX.Element} 补充分组。
 */
function SupplementSection({ title, children }) {
  return (
    <Stack spacing={1}>
      {/* 标题左侧的品牌蓝小竖条用 borderLeft 实现：随文字行高自适应，
          压低 lineHeight 让竖条保持约 14px 的短线观感。 */}
      <Typography
        sx={{
          fontSize: 13,
          fontWeight: 700,
          color: '#3c4a66',
          lineHeight: 1.2,
          borderLeft: '3px solid #2b4ecb',
          pl: 1,
        }}
      >
        {title}
      </Typography>
      {children}
    </Stack>
  );
}
