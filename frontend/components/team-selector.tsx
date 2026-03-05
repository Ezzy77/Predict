"use client"

import type { ApiTeam } from "@/lib/api"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface TeamSelectorProps {
  label: string
  teams: ApiTeam[]
  selectedTeamId: string
  onTeamChange: (teamId: string) => void
  excludeTeamId?: string
}

export function TeamSelector({
  label,
  teams,
  selectedTeamId,
  onTeamChange,
  excludeTeamId,
}: TeamSelectorProps) {
  const selectedTeam = teams.find((t) => String(t.id) === selectedTeamId)

  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </label>
      <Select value={selectedTeamId} onValueChange={onTeamChange}>
        <SelectTrigger className="h-12 w-full bg-secondary text-foreground border-border">
          <SelectValue placeholder="Select team" />
        </SelectTrigger>
        <SelectContent>
          {teams
            .filter((t) => String(t.id) !== excludeTeamId)
            .sort((a, b) => (a.stats?.position ?? 99) - (b.stats?.position ?? 99))
            .map((team) => (
              <SelectItem key={String(team.id)} value={String(team.id)}>
                <span className="flex items-center gap-2">
                  <span
                    className="inline-block h-3 w-3 rounded-full"
                    style={{
                      backgroundColor:
                        team.stats?.primary_color ?? "hsl(var(--primary))",
                    }}
                  />
                  <span>{team.name}</span>
                  {team.stats?.position != null && (
                    <span className="text-muted-foreground text-xs ml-1">
                      {formatOrdinal(team.stats.position)}
                    </span>
                  )}
                </span>
              </SelectItem>
            ))}
        </SelectContent>
      </Select>
      {selectedTeam && <TeamBrief team={selectedTeam} />}
    </div>
  )
}

function TeamBrief({ team }: { team: ApiTeam }) {
  const stats = team.stats
  if (!stats) return null

  const formLetters = stats.form ? stats.form.split("") : []
  const color = stats.primary_color ?? "hsl(var(--primary))"

  return (
    <div className="mt-2 flex items-center gap-3">
      <div
        className="flex h-10 w-10 items-center justify-center rounded-lg text-xs font-bold"
        style={{
          backgroundColor: color,
          color: getContrastColor(color),
        }}
      >
        {team.short_name}
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-xs text-muted-foreground">
          {stats.position != null && `${formatOrdinal(stats.position)} | `}
          {stats.wins != null && `${stats.wins}W ${stats.draws ?? 0}D ${stats.losses ?? 0}L`}
          {stats.points != null && ` | ${stats.points} pts`}
        </span>
        {formLetters.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Form:</span>
            {formLetters.map((ch, i) => (
              <span
                key={i}
                className={`inline-flex h-4 w-4 items-center justify-center rounded-sm text-[10px] font-bold ${
                  ch === "W"
                    ? "bg-accent text-accent-foreground"
                    : ch === "D"
                      ? "bg-muted text-muted-foreground"
                      : "bg-destructive/20 text-destructive"
                }`}
              >
                {ch}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function formatOrdinal(n: number): string {
  const s = ["th", "st", "nd", "rd"]
  const v = n % 100
  return n + (s[(v - 20) % 10] || s[v] || s[0])
}

function getContrastColor(hex: string): string {
  if (!hex.startsWith("#")) return "#FFFFFF"
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return luminance > 0.5 ? "#000000" : "#FFFFFF"
}
