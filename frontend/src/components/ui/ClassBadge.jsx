import { CLASS_STYLES, classMeaning } from '../../lib/denr'
import Tip from './Tip'

// DENR class badge in the LLDA legend colors, with an explanation on hover.
export default function ClassBadge({ param, cls }) {
  if (!cls) return <span className="font-mono text-[10px] text-faint">—</span>
  return (
    <Tip text={classMeaning(param, cls)} underline={false}>
      <span
        className={`inline-flex min-w-[46px] justify-center rounded-[2px] border px-1.5 py-px font-mono text-[10px] font-semibold ${CLASS_STYLES[cls]}`}
      >
        {cls === 'Failed' ? 'FAILED' : `CLASS ${cls}`}
      </span>
    </Tip>
  )
}
