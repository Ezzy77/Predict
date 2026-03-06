"use client"

import { useState } from "react"
import type { ApiPrediction, ApiTeam } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ChevronDown, ChevronUp } from "lucide-react"

interface PredictionDisplayProps {
  prediction: ApiPrediction
  homeTeam: ApiTeam
  awayTeam: ApiTeam
}

export function PredictionDisplay({
  prediction,
  homeTeam,
  awayTeam,
}: PredictionDisplayProps) {
  const [showBreakdown, setShowBreakdown] = useState(false)
  const homeColor = homeTeam.stats?.primary_color ?? "#6d28d9"
  const awayColor = awayTeam.stats?.primary_color ?? "#2dd4a8"
  const hasPoisson =
    prediction.poisson_home_prob != null &&
    prediction.poisson_draw_prob != null &&
    prediction.poisson_away_prob != null

  const outcomes = [
    { label: "Home Win", value: prediction.home_win_prob },
    { label: "Draw", value: prediction.draw_prob },
    { label: "Away Win", value: prediction.away_win_prob },
  ]
  const maxOutcome = outcomes.reduce((a, b) =>
    a.value > b.value ? a : b
  )

  return (
    <div className="flex flex-col gap-6">
      {/* Main Prediction Hero */}
      <Card className="border-primary/20 bg-card overflow-hidden">
        <div
          className="h-1"
          style={{
            background: `linear-gradient(to right, ${homeColor}, ${awayColor})`,
          }}
        />
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Match Prediction
            </CardTitle>
            <Badge
              variant="outline"
              className="text-xs font-mono border-primary/30 text-primary"
            >
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between gap-4">
            {/* Home Team */}
            <div className="flex flex-col items-center gap-2 flex-1">
              <div
                className="flex h-14 w-14 items-center justify-center rounded-xl text-sm font-bold shadow-lg"
                style={{
                  backgroundColor: homeColor,
                  color: getContrastColor(homeColor),
                }}
              >
                {homeTeam.short_name}
              </div>
              <span className="text-xs text-muted-foreground text-center">
                {homeTeam.name}
              </span>
              <span className="text-3xl font-bold font-mono text-foreground">
                {prediction.home_expected_goals.toFixed(1)}
              </span>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Expected Goals
              </span>
            </div>

            {/* VS / Total */}
            <div className="flex flex-col items-center gap-1">
              <span className="text-xs text-muted-foreground uppercase tracking-wider">
                Total
              </span>
              <div className="flex h-20 w-20 items-center justify-center rounded-full border-2 border-primary/30 bg-primary/5">
                <span className="text-4xl font-bold font-mono text-primary">
                  {prediction.expected_goals.toFixed(1)}
                </span>
              </div>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Predicted Goals
              </span>
            </div>

            {/* Away Team */}
            <div className="flex flex-col items-center gap-2 flex-1">
              <div
                className="flex h-14 w-14 items-center justify-center rounded-xl text-sm font-bold shadow-lg"
                style={{
                  backgroundColor: awayColor,
                  color: getContrastColor(awayColor),
                }}
              >
                {awayTeam.short_name}
              </div>
              <span className="text-xs text-muted-foreground text-center">
                {awayTeam.name}
              </span>
              <span className="text-3xl font-bold font-mono text-foreground">
                {prediction.away_expected_goals.toFixed(1)}
              </span>
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                Expected Goals
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Outcome Probabilities */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-foreground">
            Outcome Probabilities
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            {outcomes.map((o) => {
              const isTop = o === maxOutcome
              return (
                <div
                  key={o.label}
                  className={`flex flex-col items-center gap-1 rounded-lg p-3 ${
                    isTop
                      ? "bg-primary/10 border border-primary/20"
                      : "bg-secondary"
                  }`}
                >
                  <span className="text-xs text-muted-foreground">
                    {o.label}
                  </span>
                  <span
                    className={`text-2xl font-bold font-mono ${
                      isTop ? "text-primary" : "text-foreground"
                    }`}
                  >
                    {(o.value * 100).toFixed(1)}%
                  </span>
                </div>
              )
            })}
          </div>

          {/* Probability bar */}
          <div className="mt-4 flex h-3 rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${prediction.home_win_prob * 100}%`,
                backgroundColor: homeColor,
              }}
            />
            <div
              className="h-full transition-all duration-500 bg-muted-foreground/40"
              style={{
                width: `${prediction.draw_prob * 100}%`,
              }}
            />
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${prediction.away_win_prob * 100}%`,
                backgroundColor: awayColor,
              }}
            />
          </div>
          <div className="mt-1 flex items-center justify-between text-[10px] text-muted-foreground">
            <span>{homeTeam.short_name}</span>
            <span>Draw</span>
            <span>{awayTeam.short_name}</span>
          </div>

          {hasPoisson && (
            <div className="mt-4 pt-3 border-t border-border/50">
              <button
                type="button"
                onClick={() => setShowBreakdown(!showBreakdown)}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                {showBreakdown ? (
                  <ChevronUp className="h-3 w-3" />
                ) : (
                  <ChevronDown className="h-3 w-3" />
                )}
                Model breakdown (Dixon-Coles component)
              </button>
              {showBreakdown && (
                <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
                  <span>
                    Poisson: H {(prediction.poisson_home_prob! * 100).toFixed(1)}% · D{" "}
                    {(prediction.poisson_draw_prob! * 100).toFixed(1)}% · A{" "}
                    {(prediction.poisson_away_prob! * 100).toFixed(1)}%
                  </span>
                  <span className="opacity-70">
                    Blended (60% XGB + 40% Poisson): H {(prediction.home_win_prob * 100).toFixed(1)}% · D{" "}
                    {(prediction.draw_prob * 100).toFixed(1)}% · A{" "}
                    {(prediction.away_win_prob * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function getContrastColor(hex: string): string {
  if (!hex.startsWith("#")) return "#FFFFFF"
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return luminance > 0.5 ? "#000000" : "#FFFFFF"
}
