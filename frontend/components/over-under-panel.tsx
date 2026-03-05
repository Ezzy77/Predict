"use client"

import type { ApiPrediction } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowUp, ArrowDown } from "lucide-react"

interface OverUnderPanelProps {
  prediction: ApiPrediction
}

const LINES: ("1.5" | "2.5" | "3.5")[] = ["1.5", "2.5", "3.5"]

export function OverUnderPanel({ prediction }: OverUnderPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium text-foreground">
          Over / Under Probabilities
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-2">
          {LINES.map((line) => {
            const data = prediction.over_under[line]
            if (!data) return null

            const overPct = Math.round(data.over * 1000) / 10
            const underPct = Math.round(data.under * 1000) / 10
            const isOverFavored = data.over > data.under

            return (
              <div
                key={line}
                className="flex items-center gap-3 rounded-lg bg-secondary p-3"
              >
                {/* Line */}
                <div className="flex flex-col items-center min-w-[40px]">
                  <span className="text-lg font-bold font-mono text-foreground">
                    {line}
                  </span>
                  <span className="text-[10px] text-muted-foreground uppercase">
                    Goals
                  </span>
                </div>

                {/* Bar — Under (left) | Over (right) side by side */}
                <div className="flex-1 flex flex-col gap-1">
                  <div className="flex h-3 rounded-full overflow-hidden flex-row">
                    <div
                      className="h-full transition-all duration-500 shrink-0"
                      style={{
                        width: `${underPct}%`,
                        backgroundColor: "#b44aff",
                      }}
                    />
                    <div
                      className="h-full transition-all duration-500 shrink-0"
                      style={{
                        width: `${overPct}%`,
                        backgroundColor: "#2dd4a8",
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-0.5 text-xs font-mono">
                      <ArrowDown
                        className="h-3 w-3"
                        style={{ color: "#b44aff" }}
                      />
                      <span style={{ color: "#b44aff" }}>
                        {underPct.toFixed(1)}%
                      </span>
                    </span>
                    <span className="flex items-center gap-0.5 text-xs font-mono">
                      <span style={{ color: "#2dd4a8" }}>
                        {overPct.toFixed(1)}%
                      </span>
                      <ArrowUp
                        className="h-3 w-3"
                        style={{ color: "#2dd4a8" }}
                      />
                    </span>
                  </div>
                </div>

                {/* Verdict */}
                <Badge
                  variant="outline"
                  className={`text-[10px] font-mono min-w-[52px] justify-center ${
                    isOverFavored
                      ? "border-accent/40 text-accent"
                      : "border-primary/40 text-primary"
                  }`}
                  title="Model recommendation"
                >
                  {isOverFavored ? "OVER" : "UNDER"}
                </Badge>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
