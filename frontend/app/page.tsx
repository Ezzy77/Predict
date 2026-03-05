"use client"

import { useState, useEffect, useCallback } from "react"
import {
  fetchTeams,
  fetchPrediction,
  fetchPredictionBatch,
  checkHealth,
  LEAGUES,
  type ApiTeam,
  type ApiPrediction,
  type FixtureRequest,
  type LeagueCode,
} from "@/lib/api"
import { TeamSelector } from "@/components/team-selector"
import { PredictionDisplay } from "@/components/prediction-display"
import { OverUnderPanel } from "@/components/over-under-panel"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ThemeToggle } from "@/components/theme-toggle"
import {
  Activity,
  ArrowRightLeft,
  Brain,
  Info,
  Loader2,
  AlertCircle,
  CalendarDays,
  Database,
  TrendingUp,
  Wifi,
  WifiOff,
  History,
  Layers,
  AlertTriangle,
  X,
  Trash2,
  Copy,
} from "lucide-react"

const RECENT_KEY = "goalcast_recent_predictions"
const MAX_RECENT = 10

interface RecentEntry {
  home: string
  away: string
  league?: LeagueCode
  oddsHome?: string
  oddsDraw?: string
  oddsAway?: string
  oddsOver25?: string
  matchDate?: string
}

// ── Odds input sub-component ──────────────────────────────────────────────────

interface OddsInputsProps {
  oddsHome: string
  oddsDraw: string
  oddsAway: string
  oddsOver25: string
  onChange: (field: "oddsHome" | "oddsDraw" | "oddsAway" | "oddsOver25", value: string) => void
}

function OddsInputs({ oddsHome, oddsDraw, oddsAway, oddsOver25, onChange }: OddsInputsProps) {
  return (
    <Card className="border-border/40 bg-muted/20">
      <CardContent className="pt-4 pb-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            Bookmaker Odds (optional — improves accuracy)
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {(
            [
              { label: "Home Win", field: "oddsHome", value: oddsHome, placeholder: "e.g. 2.10" },
              { label: "Draw", field: "oddsDraw", value: oddsDraw, placeholder: "e.g. 3.40" },
              { label: "Away Win", field: "oddsAway", value: oddsAway, placeholder: "e.g. 3.60" },
              { label: "Over 2.5", field: "oddsOver25", value: oddsOver25, placeholder: "e.g. 1.85" },
            ] as const
          ).map(({ label, field, value, placeholder }) => (
            <div key={field} className="flex flex-col gap-1.5">
              <Label htmlFor={field} className="text-xs text-muted-foreground">
                {label}
              </Label>
              <Input
                id={field}
                type="number"
                min="1.01"
                step="0.01"
                placeholder={placeholder}
                value={value}
                onChange={(e) => onChange(field, e.target.value)}
                className="h-8 text-sm font-mono"
                aria-label={label}
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// ── Value vs odds comparison ───────────────────────────────────────────────────

function ValueVsOdds({
  prediction,
  oddsHome,
  oddsDraw,
  oddsAway,
}: {
  prediction: ApiPrediction
  oddsHome: string
  oddsDraw: string
  oddsAway: string
}) {
  const oh = parseFloat(oddsHome)
  const od = parseFloat(oddsDraw)
  const oa = parseFloat(oddsAway)
  if (!(oh > 1 && od > 1 && oa > 1)) return null

  const modelImpliedH = prediction.home_win_prob > 0.01 ? 1 / prediction.home_win_prob : 999
  const modelImpliedD = prediction.draw_prob > 0.01 ? 1 / prediction.draw_prob : 999
  const modelImpliedA = prediction.away_win_prob > 0.01 ? 1 / prediction.away_win_prob : 999

  const valueH = modelImpliedH > oh ? "value" : null
  const valueD = modelImpliedD > od ? "value" : null
  const valueA = modelImpliedA > oa ? "value" : null

  const hasValue = valueH || valueD || valueA
  if (!hasValue) return null

  return (
    <Card className="border-green-500/30 bg-green-500/5">
      <CardContent className="pt-4 pb-4">
        <div className="flex items-center gap-2 mb-2">
          <TrendingUp className="h-3.5 w-3.5 text-green-600" />
          <span className="text-xs font-medium text-green-700 dark:text-green-400 uppercase tracking-wide">
            Potential value (model odds vs bookmaker)
          </span>
        </div>
        <div className="flex flex-wrap gap-3 text-sm">
          {valueH && (
            <span className="font-mono">
              Home: model {modelImpliedH.toFixed(2)} vs {oh.toFixed(2)}
            </span>
          )}
          {valueD && (
            <span className="font-mono">
              Draw: model {modelImpliedD.toFixed(2)} vs {od.toFixed(2)}
            </span>
          )}
          {valueA && (
            <span className="font-mono">
              Away: model {modelImpliedA.toFixed(2)} vs {oa.toFixed(2)}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ── Recent predictions (localStorage) ──────────────────────────────────────────

function loadRecent(): RecentEntry[] {
  try {
    const s = localStorage.getItem(RECENT_KEY)
    if (!s) return []
    return JSON.parse(s)
  } catch {
    return []
  }
}

function saveRecent(entry: RecentEntry) {
  let recent = loadRecent()
  recent = [entry, ...recent.filter((r) => !(r.home === entry.home && r.away === entry.away && r.league === entry.league))]
  recent = recent.slice(0, MAX_RECENT)
  localStorage.setItem(RECENT_KEY, JSON.stringify(recent))
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [league, setLeague] = useState<LeagueCode>("E0")
  const [teams, setTeams] = useState<ApiTeam[]>([])
  const [teamsLoading, setTeamsLoading] = useState(true)
  const [teamsError, setTeamsError] = useState<string | null>(null)

  const [homeTeamId, setHomeTeamId] = useState<string>("")
  const [awayTeamId, setAwayTeamId] = useState<string>("")

  const [matchDate, setMatchDate] = useState<string>("")
  const [oddsHome, setOddsHome] = useState("")
  const [oddsDraw, setOddsDraw] = useState("")
  const [oddsAway, setOddsAway] = useState("")
  const [oddsOver25, setOddsOver25] = useState("")

  const [prediction, setPrediction] = useState<ApiPrediction | null>(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState<string | null>(null)

  const [batchFixtures, setBatchFixtures] = useState<RecentEntry[]>([])
  const [batchPredictions, setBatchPredictions] = useState<ApiPrediction[]>([])
  const [batchLoading, setBatchLoading] = useState(false)

  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)

  // ── Health check ───────────────────────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false))
  }, [])

  // ── Load teams when league changes ──────────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    setTeamsLoading(true)
    setTeamsError(null)
    fetchTeams(league)
      .then((data) => {
        if (cancelled) return
        setTeams(data)
        // Reset selection if current teams not in new league
        setHomeTeamId((prev) => (data.some((t) => t.id === prev) ? prev : data[0]?.id ?? ""))
        setAwayTeamId((prev) => (data.some((t) => t.id === prev) ? prev : data[1]?.id ?? ""))
      })
      .catch((err) => {
        if (cancelled) return
        setTeamsError(err.message ?? "Failed to load teams")
      })
      .finally(() => {
        if (!cancelled) setTeamsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [league])

  const buildOptions = useCallback((): Omit<FixtureRequest, "home_team" | "away_team"> => {
    const opts: Omit<FixtureRequest, "home_team" | "away_team"> = { league }
    if (matchDate) opts.match_date = matchDate
    if (parseFloat(oddsHome) > 1) opts.odds_home = parseFloat(oddsHome)
    if (parseFloat(oddsDraw) > 1) opts.odds_draw = parseFloat(oddsDraw)
    if (parseFloat(oddsAway) > 1) opts.odds_away = parseFloat(oddsAway)
    if (parseFloat(oddsOver25) > 1) opts.odds_over25 = parseFloat(oddsOver25)
    return opts
  }, [league, matchDate, oddsHome, oddsDraw, oddsAway, oddsOver25])

  const loadPrediction = useCallback(async () => {
    if (!homeTeamId || !awayTeamId || homeTeamId === awayTeamId) {
      setPrediction(null)
      return
    }
    setPredictionLoading(true)
    setPredictionError(null)
    try {
      const result = await fetchPrediction(homeTeamId, awayTeamId, buildOptions())
      setPrediction(result)
      saveRecent({
        home: homeTeamId,
        away: awayTeamId,
        league,
        oddsHome,
        oddsDraw,
        oddsAway,
        oddsOver25,
        matchDate,
      })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Prediction failed"
      setPredictionError(message)
      setPrediction(null)
    } finally {
      setPredictionLoading(false)
    }
  }, [homeTeamId, awayTeamId, buildOptions])

  const handleSwapTeams = () => {
    setHomeTeamId(awayTeamId)
    setAwayTeamId(homeTeamId)
  }

  const handleRecentClick = (entry: RecentEntry) => {
    setHomeTeamId(entry.home)
    setAwayTeamId(entry.away)
    if (entry.league) setLeague(entry.league)
    if (entry.oddsHome !== undefined) setOddsHome(entry.oddsHome)
    if (entry.oddsDraw !== undefined) setOddsDraw(entry.oddsDraw)
    if (entry.oddsAway !== undefined) setOddsAway(entry.oddsAway)
    if (entry.oddsOver25 !== undefined) setOddsOver25(entry.oddsOver25)
    if (entry.matchDate !== undefined) setMatchDate(entry.matchDate)
  }

  const runBatchPredictions = async () => {
    if (batchFixtures.length === 0) return
    setBatchLoading(true)
    try {
      const opts = buildOptions()
      const fixtures: FixtureRequest[] = batchFixtures.map((f) => ({
        ...opts,
        home_team: f.home,
        away_team: f.away,
        league: f.league ?? opts.league ?? league,
      }))
      const results = await fetchPredictionBatch(fixtures)
      setBatchPredictions(results)
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : "Batch prediction failed")
    } finally {
      setBatchLoading(false)
    }
  }

  const addToBatch = () => {
    if (homeTeamId && awayTeamId && homeTeamId !== awayTeamId) {
      setBatchFixtures((prev) => [
        ...prev,
        {
          home: homeTeamId,
          away: awayTeamId,
          league,
          oddsHome,
          oddsDraw,
          oddsAway,
          oddsOver25,
          matchDate,
        },
      ])
    }
  }

  const removeFromBatch = (index: number) => {
    setBatchFixtures((prev) => prev.filter((_, i) => i !== index))
  }

  const clearBatch = () => {
    setBatchFixtures([])
    setBatchPredictions([])
  }

  const copyBatchResults = () => {
    const text = batchPredictions
      .map((p) => `${p.home_team} vs ${p.away_team}: ${p.outcome_label} (${(p.confidence * 100).toFixed(1)}%) | Exp Goals: ${p.expected_goals.toFixed(2)}`)
      .join("\n")
    navigator.clipboard.writeText(text)
  }

  const homeTeam = teams.find((t) => t.id === homeTeamId)
  const awayTeam = teams.find((t) => t.id === awayTeamId)
  const oddsAreProvided =
    parseFloat(oddsHome) > 1 && parseFloat(oddsDraw) > 1 && parseFloat(oddsAway) > 1
  const isLowConfidence = prediction && prediction.confidence < 0.35
  const recent = loadRecent()

  return (
    <main className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
              <Activity className="h-5 w-5 text-primary" aria-hidden />
            </div>
            <div>
              <h1 className="text-lg font-bold text-foreground tracking-tight">ElevenScore AI</h1>
              <p className="text-xs text-muted-foreground">Big 5 Leagues Predictor</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            {backendOnline !== null && (
              <Badge
                variant="outline"
                className={`text-xs hidden sm:flex items-center gap-1 ${backendOnline
                  ? "border-green-500/30 text-green-600"
                  : "border-destructive/30 text-destructive"
                  }`}
              >
                {backendOnline ? (
                  <><Wifi className="h-3 w-3" aria-hidden /> API Online</>
                ) : (
                  <><WifiOff className="h-3 w-3" aria-hidden /> API Offline</>
                )}
              </Badge>
            )}
            <Badge
              variant="outline"
              className="text-xs font-mono border-primary/30 text-primary hidden sm:flex"
            >
              <Brain className="h-3 w-3 mr-1" aria-hidden />
              XGBoost
            </Badge>
            <Badge variant="secondary" className="text-xs">2025-26</Badge>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 py-6 flex flex-col gap-5">
        {backendOnline === false && (
          <Card className="border-destructive/30">
            <CardContent className="pt-5 pb-5">
              <div className="flex items-center gap-3 text-destructive">
                <AlertCircle className="h-5 w-5 shrink-0" aria-hidden />
                <div className="flex flex-col gap-0.5">
                  <span className="text-sm font-medium">Cannot reach the backend</span>
                  <span className="text-xs text-muted-foreground">
                    Run: <code className="font-mono bg-muted px-1 rounded">uvicorn main:app --reload --port 8000</code>
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {teamsError && (
          <Card className="border-destructive/30">
            <CardContent className="pt-5">
              <div className="flex items-center gap-3 text-destructive">
                <AlertCircle className="h-5 w-5 shrink-0" aria-hidden />
                <span className="text-sm">{teamsError}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {teamsLoading && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col gap-4 py-6">
                <div className="flex gap-4">
                  <Skeleton className="h-24 flex-1 rounded-lg" />
                  <Skeleton className="h-10 w-10 rounded-full shrink-0" />
                  <Skeleton className="h-24 flex-1 rounded-lg" />
                </div>
                <Skeleton className="h-10 w-32 rounded-md" />
              </div>
            </CardContent>
          </Card>
        )}

        {!teamsLoading && teams.length > 0 && (
          <>
            <Card className="bg-card/80 backdrop-blur-sm">
              <CardContent className="pt-6 flex flex-col gap-4">
                <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                  <div className="flex flex-col gap-1.5">
                    <Label className="text-xs text-muted-foreground">League</Label>
                    <Select value={league} onValueChange={(v) => setLeague(v as LeagueCode)}>
                      <SelectTrigger className="w-[180px] h-8">
                        <SelectValue placeholder="Select league" />
                      </SelectTrigger>
                      <SelectContent>
                        {LEAGUES.map((l) => (
                          <SelectItem key={l.code} value={l.code}>
                            {l.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-start">
                  <TeamSelector
                    label="Home Team"
                    teams={teams}
                    selectedTeamId={homeTeamId}
                    onTeamChange={setHomeTeamId}
                    excludeTeamId={awayTeamId}
                  />
                  <div className="flex items-center justify-center md:mt-8">
                    <button
                      onClick={handleSwapTeams}
                      className="flex h-10 w-10 items-center justify-center rounded-full border border-border bg-secondary text-muted-foreground transition-colors hover:bg-primary/10 hover:text-primary hover:border-primary/30"
                      aria-label="Swap home and away teams"
                    >
                      <ArrowRightLeft className="h-4 w-4" aria-hidden />
                    </button>
                  </div>
                  <TeamSelector
                    label="Away Team"
                    teams={teams}
                    selectedTeamId={awayTeamId}
                    onTeamChange={setAwayTeamId}
                    excludeTeamId={homeTeamId}
                  />
                </div>

                {recent.length > 0 && (
                  <div className="flex flex-wrap gap-2 items-center">
                    <History className="h-3 w-3 text-muted-foreground" aria-hidden />
                    <span className="text-xs text-muted-foreground">Recent:</span>
                    {recent.slice(0, 5).map((r, i) => {
                      const leagueName = r.league ? LEAGUES.find((l) => l.code === r.league)?.name : null
                      return (
                        <button
                          key={i}
                          type="button"
                          onClick={() => handleRecentClick(r)}
                          className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80 transition-colors flex items-center gap-1"
                        >
                          {r.home} vs {r.away}
                          {leagueName && (
                            <span className="text-[10px] opacity-70">({leagueName})</span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                )}

                <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
                  <div className="flex flex-col gap-1.5 w-full sm:w-48">
                    <Label htmlFor="match-date" className="text-xs text-muted-foreground flex items-center gap-1">
                      <CalendarDays className="h-3 w-3" aria-hidden />
                      Match Date (optional)
                    </Label>
                    <Input
                      id="match-date"
                      type="date"
                      value={matchDate}
                      onChange={(e) => setMatchDate(e.target.value)}
                      className="h-8 text-sm"
                      aria-label="Match date"
                    />
                  </div>
                </div>

                <OddsInputs
                    oddsHome={oddsHome}
                    oddsDraw={oddsDraw}
                    oddsAway={oddsAway}
                    oddsOver25={oddsOver25}
                    onChange={(field, value) => {
                      if (field === "oddsHome") setOddsHome(value)
                      if (field === "oddsDraw") setOddsDraw(value)
                      if (field === "oddsAway") setOddsAway(value)
                      if (field === "oddsOver25") setOddsOver25(value)
                    }}
                  />

                <Button
                  onClick={() => loadPrediction()}
                  disabled={
                    !homeTeamId ||
                    !awayTeamId ||
                    homeTeamId === awayTeamId ||
                    predictionLoading
                  }
                  size="lg"
                  className="w-full sm:w-auto"
                >
                  {predictionLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                      Running prediction...
                    </>
                  ) : (
                    "Get Prediction"
                  )}
                </Button>
              </CardContent>
            </Card>

            <Tabs defaultValue="single">
              <TabsList className="grid w-full grid-cols-2 max-w-md mt-4">
                <TabsTrigger value="single">Single Match</TabsTrigger>
                <TabsTrigger value="batch">Batch</TabsTrigger>
              </TabsList>

              <TabsContent value="single" className="mt-2" />

              <TabsContent value="batch" className="mt-4">
                <Card>
                  <CardContent className="pt-6 flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Layers className="h-4 w-4 text-muted-foreground" aria-hidden />
                        <span className="text-sm font-medium">Batch Queue</span>
                      </div>
                      {batchFixtures.length > 0 && (
                        <Button variant="ghost" size="sm" onClick={clearBatch} className="h-7 text-xs text-muted-foreground hover:text-destructive">
                          <Trash2 className="h-3 w-3 mr-1" />
                          Clear All
                        </Button>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Button onClick={addToBatch} disabled={!homeTeamId || !awayTeamId || homeTeamId === awayTeamId} size="sm">
                        Add current match
                      </Button>
                      <Button
                        onClick={runBatchPredictions}
                        disabled={batchFixtures.length === 0 || batchLoading}
                        size="sm"
                        variant="secondary"
                      >
                        {batchLoading ? (
                          <><Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Predicting...</>
                        ) : (
                          `Predict ${batchFixtures.length} fixture(s)`
                        )}
                      </Button>
                      {batchPredictions.length > 0 && (
                        <Button variant="outline" size="sm" onClick={copyBatchResults}>
                          <Copy className="h-3.5 w-3.5 mr-1" />
                          Copy Results
                        </Button>
                      )}
                    </div>

                    {batchFixtures.length > 0 && (
                      <div className="rounded-md border border-border bg-muted/20">
                        <ul className="divide-y divide-border">
                          {batchFixtures.map((f, i) => {
                            const fLeagueName = f.league ? LEAGUES.find((l) => l.code === f.league)?.name : null
                            return (
                            <li key={i} className="px-3 py-2 flex items-center justify-between group">
                              <div className="flex flex-col">
                                <span className="text-sm font-mono font-medium">{f.home} vs {f.away}</span>
                                <span className="text-[10px] text-muted-foreground">
                                  {fLeagueName && `${fLeagueName}`}
                                  {(f.oddsHome || f.matchDate) && fLeagueName && " · "}
                                  {f.matchDate && `Date: ${f.matchDate}`}
                                  {f.oddsHome && ` Odds: ${f.oddsHome}/${f.oddsDraw}/${f.oddsAway}`}
                                </span>
                              </div>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => removeFromBatch(i)}
                                className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/10 hover:text-destructive"
                              >
                                <X className="h-3.5 w-3.5" />
                              </Button>
                            </li>
                            )
                          })}
                        </ul>
                      </div>
                    )}

                    {batchPredictions.length > 0 && (
                      <div className="flex flex-col gap-3 mt-4 border-t pt-4">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Results</h4>
                        <div className="grid grid-cols-1 gap-2">
                          {batchPredictions.map((p, i) => (
                            <div key={i} className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg border border-border bg-card/50">
                              <div className="flex flex-col">
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-bold">{p.home_team} vs {p.away_team}</span>
                                  <Badge variant={p.confidence > 0.45 ? "default" : "secondary"} className="text-[10px] py-0 h-4">
                                    {p.outcome_label}
                                  </Badge>
                                </div>
                                <span className="text-[10px] text-muted-foreground">
                                  {(p.confidence * 100).toFixed(1)}% Confidence · {p.expected_goals.toFixed(2)} Exp Goals
                                </span>
                              </div>
                              <div className="flex items-center gap-4 mt-2 sm:mt-0">
                                <div className="flex flex-col items-center">
                                  <span className="text-[10px] text-muted-foreground uppercase">H / D / A</span>
                                  <span className="text-xs font-mono">
                                    {(p.home_win_prob * 100).toFixed(0)}% / {(p.draw_prob * 100).toFixed(0)}% / {(p.away_win_prob * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="flex flex-col items-center border-l pl-4">
                                  <span className="text-[10px] text-muted-foreground uppercase">O 2.5</span>
                                  <span className={`text-xs font-mono font-bold ${p.over_under["2.5"].over > 0.5 ? "text-green-500" : "text-muted-foreground"}`}>
                                    {(p.over_under["2.5"].over * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </>
        )}

        {predictionLoading && (
          <Card>
            <CardContent className="pt-6">
              <div className="flex flex-col gap-4 py-8">
                <div className="flex justify-center gap-8">
                  <Skeleton className="h-20 w-20 rounded-xl" />
                  <Skeleton className="h-24 w-24 rounded-full" />
                  <Skeleton className="h-20 w-20 rounded-xl" />
                </div>
                <div className="flex justify-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden />
                  <span className="text-sm text-muted-foreground">Running prediction model...</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {predictionError && !predictionLoading && (
          <Card className="border-destructive/30">
            <CardContent className="pt-5">
              <div className="flex items-center gap-3 text-destructive">
                <AlertCircle className="h-5 w-5 shrink-0" aria-hidden />
                <span className="text-sm">{predictionError}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {prediction && homeTeam && awayTeam && !predictionLoading && (
          <>
            {isLowConfidence && (
              <Card className="border-amber-500/30 bg-amber-500/5">
                <CardContent className="pt-4 pb-4 flex items-center gap-3">
                  <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" aria-hidden />
                  <span className="text-sm text-amber-700 dark:text-amber-400">
                    Low confidence — outcome probabilities are close. Consider more data or bookmaker odds.
                  </span>
                </CardContent>
              </Card>
            )}

            <PredictionDisplay prediction={prediction} homeTeam={homeTeam} awayTeam={awayTeam} />
            <OverUnderPanel prediction={prediction} />

            {oddsAreProvided && (
              <ValueVsOdds
                prediction={prediction}
                oddsHome={oddsHome}
                oddsDraw={oddsDraw}
                oddsAway={oddsAway}
              />
            )}

            <Card className="border-border/50">
              <CardContent className="pt-5 pb-5">
                <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                  <div className="flex items-start gap-3 flex-1">
                    <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" aria-hidden />
                    <div className="flex flex-col gap-1">
                      <span className="text-xs font-medium text-muted-foreground">About This Prediction</span>
                      <p className="text-xs text-muted-foreground/70 leading-relaxed">
                        {prediction.model_info ??
                          "ElevenScore AI uses XGBoost trained on Big 5 league match data (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). Features include recent form, goals scored/conceded, shots on target, corners, and bookmaker implied probabilities."}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 sm:border-l sm:border-border/40 sm:pl-4">
                    <Database className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" aria-hidden />
                    <div className="flex flex-col gap-1.5">
                      <span className="text-xs font-medium text-muted-foreground">Data Used</span>
                      <div className="flex flex-col gap-0.5">
                        <span className="text-xs text-muted-foreground/70">
                          Home: <strong className="text-foreground">{prediction.home_games_used ?? "—"}</strong> recent games
                        </span>
                        <span className="text-xs text-muted-foreground/70">
                          Away: <strong className="text-foreground">{prediction.away_games_used ?? "—"}</strong> recent games
                        </span>
                        <span className="text-xs text-muted-foreground/70">
                          Odds:{" "}
                          <strong className={oddsAreProvided ? "text-green-600" : "text-foreground"}>
                            {prediction.odds_provided ? "✓ included" : "not provided"}
                          </strong>
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </main>
  )
}
