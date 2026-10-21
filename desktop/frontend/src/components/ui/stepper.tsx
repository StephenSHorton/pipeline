import type { ReactNode } from 'react'
import { Check } from 'lucide-react'
import { cn } from 'cn'

export type StepperState = 'done' | 'current' | 'stopped' | 'todo'

export type StepperStep = {
  label: string
  time?: string
  icon?: ReactNode
  /** Overrides the state worked out from current/done (e.g. "stopped" for the step a job ended on). */
  state?: StepperState
}

const SR: Record<StepperState, string> = { done: ' (done)', current: ' (now)', stopped: ' (stopped)', todo: '' }

function Stepper({
  steps,
  current,
  done = false,
  size = 'default',
  className,
  'aria-label': ariaLabel,
}: {
  steps: readonly StepperStep[]
  current: number
  done?: boolean
  /** default: the playground's horizontal stepper. sm: a compact list that wraps, for many steps in a card. */
  size?: 'default' | 'sm'
  className?: string
  'aria-label'?: string
}) {
  if (size === 'sm') {
    return (
      <ol
        aria-label={ariaLabel}
        className={cn(
          'm-0 grid list-none grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-x-3 gap-y-1 p-0 text-[0.8rem]',
          className,
        )}
      >
        {steps.map((step, i) => {
          const state: StepperState = step.state ?? (done || i < current ? 'done' : i === current ? 'current' : 'todo')
          return (
            <li key={`${i}-${step.label}`} data-state={state} className="flex min-w-0 items-center gap-2 py-0.5">
              <span
                aria-hidden="true"
                className={cn(
                  'grid size-[18px] shrink-0 place-items-center rounded-full',
                  state === 'done' && 'bg-primary text-primary-foreground',
                  state === 'current' && 'bg-white shadow-[0_0_0_2px_var(--primary)]',
                  state === 'stopped' && 'bg-[var(--muted-2)]',
                  state === 'todo' && 'bg-[var(--mokei-detail-light)]',
                )}
              >
                {state === 'done' ? (
                  <Check className="size-3" strokeWidth={3} />
                ) : state === 'current' ? (
                  <span className="size-2 animate-pulse rounded-full bg-primary motion-reduce:animate-none" />
                ) : null}
              </span>
              <span className={cn('truncate', state === 'todo' ? 'text-ink-3' : 'text-ink', state === 'current' && 'font-semibold')}>
                {step.label}
                <span className="sr-only">{SR[state]}</span>
              </span>
              {step.time ? <span className="ml-auto shrink-0 text-[0.72rem] text-ink-3 tabular-nums">{step.time}</span> : null}
            </li>
          )
        })}
      </ol>
    )
  }
  return (
    <ol aria-label={ariaLabel} className={cn('m-0 mt-[11px] flex list-none p-0', className)}>
      {steps.map((step, i) => {
        const state = step.state ?? (done || i < current ? 'done' : i === current ? 'current' : 'todo')
        return (
          <li
            key={step.label}
            className="relative flex flex-1 flex-col items-center leading-[1.25]"
            data-state={state}
          >
            {i > 0 ? (
              <span
                className={cn(
                  'absolute top-[17px] right-1/2 mt-[-1.5px] h-[3px] w-full rounded-[2px] bg-[#e2e8f0]',
                  (i <= current || done) && 'bg-[#2f5fe6]',
                )}
              />
            ) : null}
            <span
              className={cn(
                'relative z-[1] grid size-[34px] place-items-center rounded-full bg-[#2f63e6] text-white shadow-[0_0_0_3px_#fff,0_3px_8px_rgba(37,99,235,0.25)]',
                state === 'current' &&
                  'my-[-3px] size-10 shadow-[0_0_0_3px_#fff,0_0_0_6px_#c9d7fb,0_4px_12px_rgba(37,99,235,0.3)]',
                (state === 'todo' || state === 'stopped') && 'bg-[#e5e9f0] text-[#64748b] shadow-[0_0_0_3px_#fff]',
              )}
            >
              {step.icon}
            </span>
            <span className="mt-[13px] text-[length:calc(14.1px*var(--fs))] font-medium whitespace-nowrap text-ink-2">
              {step.label}
            </span>
            {step.time ? (
              <span className="text-[length:calc(12.6px*var(--fs))] text-muted-foreground tabular-nums">
                {step.time}
              </span>
            ) : null}
          </li>
        )
      })}
    </ol>
  )
}

export { Stepper }
