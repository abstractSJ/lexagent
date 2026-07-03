import { memo } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Box, Chip, Link, Paper, Stack, Typography } from '@mui/material';

/**
 * 参考资料侧栏。
 *
 * 资料栏负责承载法条、案例和实务依据，最终回答只保留普通用户最需要的结论和行动建议。
 * 默认只显示标题，用户点击后再展开详情，避免把长法条和案例摘要直接塞进聊天回答。
 *
 * 使用 memo 的原因是最终回答流式期间 materials 引用不变，本面板可整体跳过高频增量渲染。
 *
 * @param {object} props 组件参数。
 * @param {{laws?: object[], web?: object[], warnings?: string[]}} props.materials 本轮资料集合。
 * @returns {JSX.Element} 资料侧栏。
 */
const MaterialsPanel = memo(function MaterialsPanel({ materials }) {
  const laws = Array.isArray(materials?.laws) ? materials.laws : [];
  const web = Array.isArray(materials?.web) ? materials.web : [];
  const warnings = Array.isArray(materials?.warnings) ? materials.warnings : [];
  const totalCount = laws.length + web.length;

  return (
    <Paper
      elevation={0}
      component="aside"
      aria-labelledby="materialsTitle"
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
      <Box sx={{ p: 2.25, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'rgba(248, 250, 252, 0.86)' }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography id="materialsTitle" component="h2" variant="h6" sx={{ fontWeight: 900 }}>
            参考资料
          </Typography>
          <Chip size="small" color={totalCount ? 'primary' : 'default'} label={`${totalCount} 条`} variant="outlined" />
        </Stack>
        <Typography color="text.secondary" sx={{ mt: 0.5, fontSize: 13 }}>
          法条和案例放在这里；点击标题展开查看详情。
        </Typography>
      </Box>

      <Stack spacing={1.5} sx={{ minHeight: 0, overflowY: 'auto', p: 2 }}>
        {totalCount === 0 && warnings.length === 0 && (
          <Paper variant="outlined" sx={{ p: 1.5, borderRadius: 1.5, backgroundColor: '#f8fafc' }}>
            <Typography color="text.secondary" sx={{ fontSize: 13, lineHeight: 1.6 }}>
              本轮暂无可展示资料。发送问题后，相关法条和案例会整理到这里。
            </Typography>
          </Paper>
        )}

        {laws.length > 0 && <MaterialGroup title="法条依据" items={laws} />}
        {web.length > 0 && <MaterialGroup title="案例 / 实务资料" items={web} />}

        {warnings.length > 0 && (
          <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 1.5, borderColor: '#fde68a', backgroundColor: '#fffbeb' }}>
            <Typography sx={{ fontSize: 13, fontWeight: 900, color: '#92400e' }}>资料提示</Typography>
            <Stack component="ul" spacing={0.5} sx={{ pl: 2, my: 0.75 }}>
              {warnings.map((warning, index) => (
                <Typography key={`${warning}-${index}`} component="li" sx={{ fontSize: 12.5, lineHeight: 1.5, color: '#92400e' }}>
                  {warning}
                </Typography>
              ))}
            </Stack>
          </Paper>
        )}
      </Stack>
    </Paper>
  );
});

export default MaterialsPanel;

/**
 * 资料分组。
 *
 * @param {object} props 组件参数。
 * @param {string} props.title 分组标题。
 * @param {object[]} props.items 分组资料。
 * @returns {JSX.Element} 资料分组。
 */
function MaterialGroup({ title, items }) {
  return (
    <Stack spacing={1}>
      <Typography sx={{ fontSize: 13, fontWeight: 900 }}>{title}</Typography>
      {items.map((item, index) => (
        <MaterialItem key={item.id || `${item.material_type}-${item.title}-${index}`} item={item} />
      ))}
    </Stack>
  );
}

/**
 * 单条可展开资料。
 *
 * @param {object} props 组件参数。
 * @param {object} props.item 资料条目。
 * @returns {JSX.Element} 可展开资料卡。
 */
function MaterialItem({ item }) {
  const chipLabel = item.material_type === 'law' ? '法条' : '资料';

  return (
    <Accordion disableGutters elevation={0} variant="outlined" sx={{ borderRadius: 1.5, overflow: 'hidden', '&:before': { display: 'none' } }}>
      <AccordionSummary
        aria-label={`查看资料：${item.title}`}
        sx={{
          minHeight: 48,
          alignItems: 'flex-start',
          '& .MuiAccordionSummary-content': { my: 1, minWidth: 0 },
        }}
      >
        <Stack spacing={0.5} sx={{ minWidth: 0, width: '100%' }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography sx={{ fontSize: 13.5, fontWeight: 900, lineHeight: 1.45, wordBreak: 'break-word' }}>
              {item.title}
            </Typography>
            <Chip size="small" label={chipLabel} variant="outlined" sx={{ flexShrink: 0 }} />
          </Stack>
          {item.subtitle && (
            <Typography color="text.secondary" sx={{ fontSize: 12.5, lineHeight: 1.45, wordBreak: 'break-word' }}>
              {item.subtitle}
            </Typography>
          )}
        </Stack>
      </AccordionSummary>
      <AccordionDetails sx={{ pt: 0, borderTop: '1px solid', borderColor: 'divider' }}>
        <Stack spacing={1}>
          {item.detail && (
            <Typography sx={{ fontSize: 13, lineHeight: 1.65, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{item.detail}</Typography>
          )}
          {item.source && (
            <Typography color="text.secondary" sx={{ fontSize: 12.5 }}>
              来源：{item.source}
            </Typography>
          )}
          {item.url && (
            <Link href={item.url} target="_blank" rel="noreferrer" sx={{ fontSize: 13, fontWeight: 700 }}>
              打开来源
            </Link>
          )}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}
