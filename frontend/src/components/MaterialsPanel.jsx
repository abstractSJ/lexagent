import { memo } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Chip, Link, Paper, Stack, Typography } from '@mui/material';
import PanelShell from './PanelShell.jsx';
import {
  AlertTriangleIcon,
  BookOpenIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  GlobeIcon,
  SearchIcon,
} from '../icons.jsx';

/**
 * 参考资料侧栏。
 *
 * 资料栏负责承载法条、案例和实务依据，最终回答只保留普通用户最需要的结论和行动建议。
 * 默认只显示标题，用户点击后再展开详情，避免把长法条和案例摘要直接塞进聊天回答。
 * 面板头部用金色系强调：金色是法律行业的经典点缀色，让"权威资料"与蓝色的对话区、
 * 紫色的进度区在三栏中一眼可分。
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
    <PanelShell
      component="aside"
      titleId="materialsTitle"
      icon={<BookOpenIcon />}
      accent={{ bg: '#fdf3e3', fg: '#b7791f' }}
      title="参考资料"
      subtitle="法条和案例放在这里；点击标题展开查看详情。"
      action={
        <Chip size="small" color={totalCount ? 'primary' : 'default'} label={`${totalCount} 条`} variant="outlined" />
      }
    >
      <Stack spacing={2} sx={{ minHeight: 0, overflowY: 'auto', p: 2 }}>
        {totalCount === 0 && warnings.length === 0 && (
          <Stack spacing={1} sx={{ py: 5, alignItems: 'center' }}>
            <SearchIcon sx={{ fontSize: 28, color: '#c3cede' }} />
            <Typography color="text.secondary" sx={{ fontSize: 13, textAlign: 'center', px: 2 }}>
              本轮暂无可展示资料。发送问题后，相关法条和案例会整理到这里。
            </Typography>
          </Stack>
        )}

        {laws.length > 0 && <MaterialGroup title="法条依据" items={laws} />}
        {web.length > 0 && <MaterialGroup title="案例 / 实务资料" items={web} />}

        {warnings.length > 0 && (
          <Paper
            variant="outlined"
            sx={{ p: 1.5, borderRadius: '12px', borderColor: '#f0dcae', backgroundColor: '#fdf7e7' }}
          >
            <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
              <AlertTriangleIcon sx={{ fontSize: 15, color: '#b7791f' }} />
              <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#9a6410' }}>资料提示</Typography>
            </Stack>
            <Stack component="ul" spacing={0.5} sx={{ pl: 2, my: 0.75 }}>
              {warnings.map((warning, index) => (
                <Typography
                  key={`${warning}-${index}`}
                  component="li"
                  sx={{ fontSize: 12.5, lineHeight: 1.5, color: '#7a5410' }}
                >
                  {warning}
                </Typography>
              ))}
            </Stack>
          </Paper>
        )}
      </Stack>
    </PanelShell>
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
  // 分组身份图标按标题判断：法条组用书本、案例/实务组用地球，与事件时间线的图标语义保持一致。
  const GroupIcon = title === '法条依据' ? BookOpenIcon : GlobeIcon;

  return (
    <Stack spacing={1}>
      <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
        <GroupIcon sx={{ fontSize: 14, color: '#8592ab' }} />
        <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#3c4a66' }}>{title}</Typography>
        <Typography color="text.secondary" sx={{ fontSize: 12 }}>{`· ${items.length}`}</Typography>
      </Stack>
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
  const isLaw = item.material_type === 'law';
  const chipLabel = isLaw ? '法条' : '资料';
  // 类型 Chip 用软底色区分资料身份：法条走品牌蓝、案例/实务走绿色，去掉边框保持轻量。
  const chipSx = isLaw
    ? { border: 'none', backgroundColor: '#e8edfb', color: '#2b4ecb', height: 20, fontSize: 11 }
    : { border: 'none', backgroundColor: '#e6f6f2', color: '#0f8a5f', height: 20, fontSize: 11 };

  return (
    <Accordion
      disableGutters
      elevation={0}
      sx={{
        borderRadius: '12px',
        border: '1px solid #e3e9f4',
        overflow: 'hidden',
        // hover/展开只加深边框，不加投影：资料列表条目多，逐条投影会显得杂乱。
        transition: 'border-color 0.18s ease',
        '&:hover': { borderColor: '#c8d4ee' },
        '&.Mui-expanded': { borderColor: '#b9c9ee' },
      }}
    >
      <AccordionSummary
        aria-label={`查看资料：${item.title}`}
        expandIcon={<ChevronDownIcon sx={{ fontSize: 16, color: '#8592ab' }} />}
        sx={{
          minHeight: 46,
          '& .MuiAccordionSummary-content': { my: 1, minWidth: 0 },
        }}
      >
        <Stack spacing={0.5} sx={{ minWidth: 0, width: '100%' }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography sx={{ fontSize: 13.5, fontWeight: 700, lineHeight: 1.45, wordBreak: 'break-word' }}>
              {item.title}
            </Typography>
            <Chip size="small" label={chipLabel} sx={{ flexShrink: 0, ...chipSx }} />
          </Stack>
          {item.subtitle && (
            <Typography color="text.secondary" sx={{ fontSize: 12.5, lineHeight: 1.45, wordBreak: 'break-word' }}>
              {item.subtitle}
            </Typography>
          )}
        </Stack>
      </AccordionSummary>
      <AccordionDetails
        sx={{ pt: 1.25, borderTop: '1px solid', borderColor: 'divider', backgroundColor: '#fbfcfe' }}
      >
        <Stack spacing={1}>
          {item.detail && (
            <Typography sx={{ fontSize: 13, lineHeight: 1.7, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {item.detail}
            </Typography>
          )}
          {item.source && (
            <Typography color="text.secondary" sx={{ fontSize: 12.5 }}>
              来源：{item.source}
            </Typography>
          )}
          {item.url && (
            <Link
              href={item.url}
              target="_blank"
              rel="noreferrer"
              sx={{ fontSize: 12.5, fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: 0.5 }}
            >
              <ExternalLinkIcon sx={{ fontSize: 13 }} />
              打开来源
            </Link>
          )}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}
