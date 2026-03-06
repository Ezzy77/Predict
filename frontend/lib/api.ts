const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

// ─────────────────────────────────────────────────────────────────────────────
// Raw backend response types (what the API actually returns)
// ─────────────────────────────────────────────────────────────────────────────

interface BackendTeamsResponse {
  teams: string[]
  count: number
}

interface BackendPrediction {
  // Outcome
  predicted_outcome: "H" | "D" | "A"
  outcome_label: "Home Win" | "Draw" | "Away Win"
  home_win_prob: number
  draw_prob: number
  away_win_prob: number
  // Goals
  expected_goals: number
  home_expected_goals?: number
  away_expected_goals?: number
  // Dixon-Coles Poisson component (for transparency)
  poisson_home_prob?: number
  poisson_draw_prob?: number
  poisson_away_prob?: number
  // Betting edges (model prob - bookmaker implied)
  edge_home?: number
  edge_draw?: number
  edge_away?: number
  over_1_5_prob: number
  over_2_5_prob: number
  under_2_5_prob: number
  over_3_5_prob: number
  // Fixture metadata (only on /predict/fixture)
  home_team?: string
  away_team?: string
  match_date?: string
  home_games_used?: number
  away_games_used?: number
  odds_provided?: boolean
  features_used?: Record<string, number>
}

// ─────────────────────────────────────────────────────────────────────────────
// Frontend-facing types (what your components consume)
// ─────────────────────────────────────────────────────────────────────────────

export interface ApiTeamStats {
  goals_scored: number
  goals_conceded: number
  matches_played: number
  form: string          // e.g. "WWDLW" — derived from recent results
  position: number      // league position (not in API, set to 0 if unknown)
  wins?: number
  draws?: number
  losses?: number
  points?: number
  primary_color?: string
  [key: string]: unknown
}

export interface ApiTeam {
  id: string            // team name used as id (what the backend accepts)
  name: string
  short_name: string    // abbreviated name for compact display
  logo_url?: string
  stats?: ApiTeamStats
}

export interface OverUnderLine {
  over: number
  under: number
}

export interface ApiPrediction {
  home_team: string
  away_team: string
  expected_goals: number
  home_expected_goals: number   // estimated split from total
  away_expected_goals: number   // estimated split from total
  home_win_prob: number
  draw_prob: number
  away_win_prob: number
  predicted_outcome: "H" | "D" | "A"
  outcome_label: string
  over_under: {
    "1.5": OverUnderLine
    "2.5": OverUnderLine
    "3.5": OverUnderLine
  }
  confidence: number            // derived: max(home, draw, away) probability
  model_info?: string
  // Dixon-Coles Poisson component
  poisson_home_prob?: number
  poisson_draw_prob?: number
  poisson_away_prob?: number
  // Betting edges (model prob - bookmaker implied); positive = value bet
  edge_home?: number
  edge_draw?: number
  edge_away?: number
  // Extras surfaced from backend
  home_games_used?: number
  away_games_used?: number
  odds_provided?: boolean
  features_used?: Record<string, number>
}

export const LEAGUES = [
  { code: "E0", name: "Premier League" },
  { code: "D1", name: "Bundesliga" },
  { code: "SP1", name: "La Liga" },
  { code: "I1", name: "Serie A" },
  { code: "F1", name: "Ligue 1" },
] as const

export type LeagueCode = (typeof LEAGUES)[number]["code"]

export interface FixtureRequest {
  home_team: string
  away_team: string
  league?: LeagueCode           // E0, D1, SP1, I1, F1 — default E0
  match_date?: string           // "YYYY-MM-DD", optional
  odds_home?: number
  odds_draw?: number
  odds_away?: number
  odds_over25?: number
  window?: number               // rolling window size, default 5
}

// ─────────────────────────────────────────────────────────────────────────────
// Abbreviation map — extend as needed
// ─────────────────────────────────────────────────────────────────────────────

const SHORT_NAME_MAP: Record<string, string> = {
  "Arsenal": "ARS",
  "Aston Villa": "AVL",
  "Bournemouth": "BOU",
  "Brentford": "BRE",
  "Brighton": "BHA",
  "Burnley": "BUR",
  "Chelsea": "CHE",
  "Crystal Palace": "CRY",
  "Everton": "EVE",
  "Fulham": "FUL",
  "Leeds": "LEE",
  "Liverpool": "LIV",
  "Man City": "MCI",
  "Man United": "MUN",
  "Newcastle": "NEW",
  "Nott'm Forest": "NFO",
  "Sunderland": "SUN",
  "Tottenham": "TOT",
  "West Ham": "WHU",
  "Wolves": "WOL",
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

const MAX_RETRIES = 3
const RETRY_DELAY_MS = 1000

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let lastError: Error | null = null
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${BASE_URL}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...init,
      })
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`
        try {
          const body = await res.json()
          if (body?.detail) detail = body.detail
        } catch { /* ignore */ }
        throw new Error(detail)
      }
      return (await res.json()) as T
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      if (attempt < MAX_RETRIES - 1) {
        await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * (attempt + 1)))
      }
    }
  }
  throw lastError ?? new Error("Request failed")
}

/**
 * Split total expected goals into home/away using win probability weighting.
 * The team more likely to win is assumed to score proportionally more.
 */
function splitExpectedGoals(
  total: number,
  homeWinProb: number,
  awayWinProb: number
): { home: number; away: number } {
  const totalProb = homeWinProb + awayWinProb || 1
  const homeShare = homeWinProb / totalProb
  const awayShare = awayWinProb / totalProb
  return {
    home: Math.round(total * homeShare * 100) / 100,
    away: Math.round(total * awayShare * 100) / 100,
  }
}

/** Transform raw backend prediction into the richer frontend ApiPrediction shape. */
function transformPrediction(
  raw: BackendPrediction,
  homeTeam: string,
  awayTeam: string
): ApiPrediction {
  const split = splitExpectedGoals(raw.expected_goals, raw.home_win_prob, raw.away_win_prob)
  const homeExp = raw.home_expected_goals ?? split.home
  const awayExp = raw.away_expected_goals ?? split.away

  const confidence = Math.max(raw.home_win_prob, raw.draw_prob, raw.away_win_prob)

  return {
    home_team: raw.home_team ?? homeTeam,
    away_team: raw.away_team ?? awayTeam,
    predicted_outcome: raw.predicted_outcome,
    outcome_label: raw.outcome_label,
    home_win_prob: raw.home_win_prob,
    draw_prob: raw.draw_prob,
    away_win_prob: raw.away_win_prob,
    expected_goals: raw.expected_goals,
    home_expected_goals: homeExp,
    away_expected_goals: awayExp,
    over_under: {
      "1.5": { over: raw.over_1_5_prob, under: 1 - raw.over_1_5_prob },
      "2.5": { over: raw.over_2_5_prob, under: raw.under_2_5_prob },
      "3.5": { over: raw.over_3_5_prob, under: 1 - raw.over_3_5_prob },
    },
    confidence: Math.round(confidence * 1000) / 1000,
    model_info: `Hybrid (XGBoost + Dixon-Coles) — ${raw.home_games_used ?? "?"}H / ${raw.away_games_used ?? "?"}A games used`,
    poisson_home_prob: raw.poisson_home_prob,
    poisson_draw_prob: raw.poisson_draw_prob,
    poisson_away_prob: raw.poisson_away_prob,
    edge_home: raw.edge_home,
    edge_draw: raw.edge_draw,
    edge_away: raw.edge_away,
    home_games_used: raw.home_games_used,
    away_games_used: raw.away_games_used,
    odds_provided: raw.odds_provided,
    features_used: raw.features_used,
  }
}

/** Transform a team name string into an ApiTeam object. */
function teamFromName(name: string): ApiTeam {
  return {
    id: name,
    name,
    short_name: SHORT_NAME_MAP[name] ?? name.slice(0, 3).toUpperCase(),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch teams. Optionally filter by league (E0, D1, SP1, I1, F1).
 * Backend: GET /teams?league=E0 → { teams: string[], count: number }
 */
export async function fetchTeams(league?: LeagueCode): Promise<ApiTeam[]> {
  const params = league ? `?league=${league}` : ""
  const data = await apiFetch<BackendTeamsResponse>(`/teams${params}`)
  return data.teams.map(teamFromName)
}

/**
 * Fetch a single team by name.
 * The backend has no individual team endpoint, so we resolve from the teams list.
 */
export async function fetchTeam(teamId: string): Promise<ApiTeam> {
  const teams = await fetchTeams()
  const found = teams.find(
    (t) => t.id === teamId || t.name.toLowerCase() === teamId.toLowerCase()
  )
  if (!found) throw new Error(`Team not found: ${teamId}`)
  return found
}

/**
 * Predict a fixture by team names.
 * Backend: POST /predict/fixture
 *
 * @param homeTeam  - exact team name (see fetchTeams)
 * @param awayTeam  - exact team name
 * @param options   - optional match date, bookmaker odds, rolling window
 */
export async function fetchPrediction(
  homeTeam: string,
  awayTeam: string,
  options?: Omit<FixtureRequest, "home_team" | "away_team">
): Promise<ApiPrediction> {
  const body: FixtureRequest = {
    home_team: homeTeam,
    away_team: awayTeam,
    ...options,
  }
  const raw = await apiFetch<BackendPrediction>("/predict/fixture", {
    method: "POST",
    body: JSON.stringify(body),
  })
  return transformPrediction(raw, homeTeam, awayTeam)
}

/**
 * Predict multiple fixtures in one request.
 * Backend: POST /predict/fixture/batch (max 50)
 */
export async function fetchPredictionBatch(
  fixtures: FixtureRequest[]
): Promise<ApiPrediction[]> {
  if (fixtures.length === 0) return []
  const raws = await apiFetch<BackendPrediction[]>("/predict/fixture/batch", {
    method: "POST",
    body: JSON.stringify(fixtures),
  })
  return raws.map((raw, i) =>
    transformPrediction(raw, fixtures[i].home_team, fixtures[i].away_team)
  )
}

/**
 * Health check — useful to verify the backend is up before rendering the UI.
 */
export async function checkHealth(): Promise<{
  status: string
  models_loaded: boolean
  historical_matches: number
  known_teams: number
}> {
  return apiFetch("/health")
}
