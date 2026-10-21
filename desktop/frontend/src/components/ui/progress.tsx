import { Progress as ProgressPrimitive } from "@base-ui/react/progress"
import { cn } from "cn"

function Progress({
  className,
  children,
  value,
  tone = "primary",
  size = "default",
  ...props
}: ProgressPrimitive.Root.Props & {
  tone?: "primary" | "blue" | "green"
  size?: "default" | "hud" | "hud-battery" | "hud-mini"
}) {
  const hud = size !== "default"
  return (
    <ProgressPrimitive.Root
      value={value}
      data-slot="progress"
      data-tone={tone}
      data-size={size}
      className={cn(hud ? "contents" : "flex flex-wrap gap-3", !hud && className)}
      {...props}
    >
      {children}
      <ProgressPrimitive.Track
        data-slot="progress-track"
        className={cn(
          hud
            ? "relative block overflow-hidden rounded-full bg-[#dee5f0]"
            : "relative flex h-1 w-full items-center overflow-x-hidden rounded-full bg-muted",
          size === "hud" && "h-[7px] w-full",
          size === "hud-battery" && "h-2 min-w-0 flex-1",
          size === "hud-mini" && "h-[5px] w-[58px]",
          hud && className,
        )}
      >
        <ProgressPrimitive.Indicator
          data-slot="progress-indicator"
          className={cn(
            hud
              ? "absolute inset-y-0 left-0 rounded-full transition-[width] duration-400 ease-in-out"
              : "h-full bg-primary transition-all",
            tone === "green" && "bg-[linear-gradient(90deg,#46d86d,#1ca542)]",
            tone === "blue" && "bg-[linear-gradient(90deg,#3f7bf0,#2a55e3)]",
          )}
        />
      </ProgressPrimitive.Track>
    </ProgressPrimitive.Root>
  )
}

function ProgressTrack({ className, ...props }: ProgressPrimitive.Track.Props) {
  return (
    <ProgressPrimitive.Track
      className={cn(
        "relative flex h-1 w-full items-center overflow-x-hidden rounded-full bg-muted",
        className
      )}
      data-slot="progress-track"
      {...props}
    />
  )
}

function ProgressIndicator({
  className,
  ...props
}: ProgressPrimitive.Indicator.Props) {
  return (
    <ProgressPrimitive.Indicator
      data-slot="progress-indicator"
      className={cn("h-full bg-primary transition-all", className)}
      {...props}
    />
  )
}

function ProgressLabel({ className, ...props }: ProgressPrimitive.Label.Props) {
  return (
    <ProgressPrimitive.Label
      className={cn("text-sm font-medium", className)}
      data-slot="progress-label"
      {...props}
    />
  )
}

function ProgressValue({ className, ...props }: ProgressPrimitive.Value.Props) {
  return (
    <ProgressPrimitive.Value
      className={cn(
        "ml-auto text-sm text-muted-foreground tabular-nums",
        className
      )}
      data-slot="progress-value"
      {...props}
    />
  )
}

export {
  Progress,
  ProgressTrack,
  ProgressIndicator,
  ProgressLabel,
  ProgressValue,
}
