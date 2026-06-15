const BENCHMARK = "^TWII";
const COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"];

const CORS_PROXIES = [
  url => `https://corsproxy.io/?url=${encodeURIComponent(url)}`,
  url => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
];

// ── Date helpers ─────────────────────────────────────────────────────────
function ymd(d) {
  return d.toISOString().slice(0, 10);
}
function dateOnly(d) {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}
function addDays(d, n) {
  const r = new Date(d);
  r.setDate(r.getDate() + n);
  return r;
}
function nowInTaipei() {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
  return new Date(utcMs + 8 * 3600000);
}
function isMarketOpen() {
  const t = nowInTaipei();
  const day = t.getUTCDay();
  if (day === 0 || day === 6) return false;
  const mins = t.getUTCHours() * 60 + t.getUTCMinutes();
  return mins >= 9 * 60 && mins <= 13 * 60 + 30;
}
function isoWeekKey(dateStr) {
  const d = new Date(dateStr + "T00:00:00");
  const target = new Date(d.valueOf());
  const dayNr = (d.getDay() + 6) % 7; // Mon=0..Sun=6
  target.setDate(target.getDate() - dayNr + 3); // Thursday of this week
  const firstThursday = new Date(target.getFullYear(), 0, 4);
  const diff = target - firstThursday;
  const week = 1 + Math.round(diff / (7 * 86400000));
  return `${target.getFullYear()}-W${week}`;
}

// ── Array helpers ────────────────────────────────────────────────────────
function ffill(arr) {
  let last = null;
  return arr.map(v => {
    if (v != null && !Number.isNaN(v)) { last = v; return v; }
    return last;
  });
}
function bfill(arr) {
  return ffill(arr.slice().reverse()).reverse();
}
function ffillBfill(arr) {
  return bfill(ffill(arr));
}
function mean(a) {
  return a.reduce((s, v) => s + v, 0) / a.length;
}
function std(a, ddof = 1) {
  const m = mean(a);
  const ss = a.reduce((s, v) => s + (v - m) ** 2, 0);
  return Math.sqrt(ss / (a.length - ddof));
}
function pctChange(arr) {
  const out = [];
  for (let i = 1; i < arr.length; i++) out.push((arr[i] - arr[i - 1]) / arr[i - 1]);
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────
async function fetchYahoo(symbol, period1, period2, interval = "1d") {
  const target = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}` +
    `?period1=${Math.floor(period1.getTime() / 1000)}&period2=${Math.floor(period2.getTime() / 1000)}` +
    `&interval=${interval}&events=div&includeAdjustedClose=true`;

  let lastErr;
  for (const proxy of CORS_PROXIES) {
    try {
      const res = await fetch(proxy(target));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      const result = json?.chart?.result?.[0];
      if (!result || !result.timestamp) throw new Error("No data");
      return result;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error("All proxies failed");
}

function parseSeries(result) {
  const dates = result.timestamp.map(t => dateOnly(new Date(t * 1000)));
  const quote = result.indicators.quote[0];
  const adj = result.indicators.adjclose?.[0]?.adjclose || quote.close;
  const close = ffillBfill(quote.close);
  const adjclose = ffillBfill(adj);
  const dividends = [];
  const divEvents = result.events?.dividends;
  if (divEvents) {
    for (const ev of Object.values(divEvents)) dividends.push({ date: new Date(ev.date * 1000), amount: ev.amount });
    dividends.sort((a, b) => a.date - b.date);
  }
  return { dates, close, adjclose, dividends };
}

async function fetchLivePrice(ticker) {
  try {
    const now = new Date();
    const result = await fetchYahoo(ticker, addDays(now, -2), addDays(now, 1), "1m");
    const quote = result.indicators.quote[0];
    const closes = quote.close.filter(v => v != null);
    if (!closes.length) return null;
    return closes[closes.length - 1];
  } catch (e) {
    return null;
  }
}

async function fetch0050Yield() {
  try {
    const end = addDays(new Date(), 1);
    const start = addDays(new Date(), -400);
    const result = await fetchYahoo("0050.TW", start, end, "1d");
    const { close, dividends } = parseSeries(result);
    const cutoff = addDays(new Date(), -365);
    const totalDiv = dividends.filter(d => d.date >= cutoff).reduce((s, d) => s + d.amount, 0);
    const price = close[close.length - 1];
    const yieldRate = price ? totalDiv / price : 0.017;
    return { yieldRate, totalDiv, price };
  } catch (e) {
    return { yieldRate: 0.017, totalDiv: 0, price: 0 };
  }
}

// ── Stats: linear regression with p-value ───────────────────────────────
function gammaln(x) {
  const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}
function betacf(a, b, x) {
  const MAXIT = 200, EPS = 3e-7, FPMIN = 1e-30;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}
function betai(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
  return 1 - bt * betacf(b, a, 1 - x) / b;
}
function tTwoSidedP(t, df) {
  const x = df / (df + t * t);
  return betai(df / 2, 0.5, x);
}
function linregress(x, y) {
  const n = x.length;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  const r = sxy / Math.sqrt(sxx * syy);
  const r2 = r * r;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const yhat = intercept + slope * x[i];
    ssRes += (y[i] - yhat) ** 2;
  }
  const df = n - 2;
  const se = Math.sqrt(ssRes / df / sxx);
  const t = slope / se;
  const pvalue = tTwoSidedP(t, df);
  return { slope, intercept, r2, se, pvalue };
}
function skewness(a) {
  const n = a.length, m = mean(a);
  const m2 = a.reduce((s, v) => s + (v - m) ** 2, 0) / n;
  const m3 = a.reduce((s, v) => s + (v - m) ** 3, 0) / n;
  const g1 = m3 / Math.pow(m2, 1.5);
  return Math.sqrt(n * (n - 1)) / (n - 2) * g1;
}
function kurtosisExcess(a) {
  const n = a.length, m = mean(a);
  const m2 = a.reduce((s, v) => s + (v - m) ** 2, 0) / n;
  const m4 = a.reduce((s, v) => s + (v - m) ** 4, 0) / n;
  const g2 = m4 / (m2 * m2) - 3;
  return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6);
}
function correlation(a, b) {
  const n = a.length, ma = mean(a), mb = mean(b);
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) {
    cov += (a[i] - ma) * (b[i] - mb);
    va += (a[i] - ma) ** 2;
    vb += (b[i] - mb) ** 2;
  }
  return cov / Math.sqrt(va * vb);
}
function maxDrawdownFromReturns(rets) {
  let cum = 1, peak = 1, minDD = 0;
  for (const r of rets) {
    cum *= (1 + r);
    if (cum > peak) peak = cum;
    const dd = (cum / peak - 1) * 100;
    if (dd < minDD) minDD = dd;
  }
  return minDD;
}

// ── Domain calculations ───────────────────────────────────────────────────
function periodRf(annualRf, freq) {
  const divisor = freq === "D" ? 252 : 52;
  return Math.pow(1 + annualRf, 1 / divisor) - 1;
}
function annualiseAlpha(alphaPeriod, freq) {
  const mult = freq === "D" ? 252 : 52;
  return (Math.pow(1 + alphaPeriod, mult) - 1) * 100;
}
function rollingMean(arr, window) {
  return arr.map((_, i) => i < window - 1 ? null : mean(arr.slice(i - window + 1, i + 1)));
}
function computeBias(dates, close) {
  const ma5 = rollingMean(close, 5);
  const ma20 = rollingMean(close, 20);
  const bias5 = close.map((c, i) => ma5[i] != null ? (c - ma5[i]) / ma5[i] * 100 : null);
  const bias20 = close.map((c, i) => ma20[i] != null ? (c - ma20[i]) / ma20[i] * 100 : null);
  return { dates, close, ma5, ma20, bias5, bias20 };
}
function biasSignal(bias, window) {
  const hi = window === 5 ? 3.0 : 5.0;
  const lo = -hi;
  if (bias > hi) return "🔴 Overbought";
  if (bias > hi / 2) return "🟡 Slightly High";
  if (bias < lo) return "🟢 Oversold";
  if (bias < lo / 2) return "🟡 Slightly Low";
  return "⚪ Neutral";
}
function cellClass(v) {
  v = String(v);
  if (["Outperforms", "Oversold", "Defensive"].some(k => v.includes(k))) return "cell-green";
  if (["Underperforms", "Overbought"].some(k => v.includes(k))) return "cell-red";
  if (v.includes("Slightly High") || v.includes("Aggressive")) return "cell-orange";
  if (v.includes("Slightly Low")) return "cell-yellow";
  if (v.includes("Neutral")) return "cell-grey";
  if (v.startsWith("+") && v.includes("%")) return "cell-green";
  if (v.startsWith("-") && v.includes("%")) return "cell-red";
  return "";
}
function corrColor(v) {
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const g = Math.round(255 * (1 - t));
    return `rgb(${g},${g},255)`;
  }
  const gb = Math.round(255 * (1 + t));
  return `rgb(255,${gb},${gb})`;
}

// ── Data alignment ───────────────────────────────────────────────────────
function alignSeries(seriesMap) {
  const tickers = Object.keys(seriesMap);
  let common = null;
  for (const tk of tickers) {
    const set = new Set(seriesMap[tk].dates.map(ymd));
    common = common ? new Set([...common].filter(d => set.has(d))) : set;
  }
  const sortedDates = [...common].sort();
  const aligned = {};
  for (const tk of tickers) {
    const m = new Map(seriesMap[tk].dates.map((d, i) => [ymd(d), seriesMap[tk].adjclose[i]]));
    aligned[tk] = sortedDates.map(d => m.get(d));
  }
  return { dates: sortedDates, values: aligned };
}
function weeklyIndices(dates) {
  const groups = new Map();
  dates.forEach((d, i) => groups.set(isoWeekKey(d), i));
  return [...groups.values()].sort((a, b) => a - b);
}

// ── Charts ───────────────────────────────────────────────────────────────
let charts = [];
function destroyCharts() {
  charts.forEach(c => c.destroy());
  charts = [];
}
function makeChart(canvas, config) {
  const c = new Chart(canvas.getContext("2d"), config);
  charts.push(c);
  return c;
}

function renderCumulativeChart(canvas, dates, aligned, tickers) {
  const datasets = tickers.map((tk, i) => {
    const vals = aligned[tk];
    const base = vals[0];
    const cr = vals.map(v => (v / base - 1) * 100);
    const isBm = tk === BENCHMARK;
    return {
      label: tk, data: cr,
      borderColor: isBm ? "#9E9E9E" : COLORS[i % 4],
      borderWidth: isBm ? 2.5 : 2,
      borderDash: isBm ? [4, 4] : [],
      pointRadius: 0, fill: false,
    };
  });
  makeChart(canvas, {
    type: "line",
    data: { labels: dates, datasets },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: "Cumulative Returns (%)" }, legend: { position: "bottom" } },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Return (%)" } },
      },
    },
  });
}

function renderScatterChart(canvas, tk, res, color) {
  const x = res.excBench.map(v => v * 100);
  const y = res.excAsset.map(v => v * 100);
  const points = x.map((xv, i) => ({ x: xv, y: y[i] }));
  const xmin = Math.min(...x), xmax = Math.max(...x);
  const fitPoints = [
    { x: xmin, y: res.alphaPeriod * 100 + res.beta * xmin },
    { x: xmax, y: res.alphaPeriod * 100 + res.beta * xmax },
  ];
  makeChart(canvas, {
    type: "scatter",
    data: {
      datasets: [
        { type: "scatter", label: "Observations", data: points, backgroundColor: color, pointRadius: 4, pointHoverRadius: 5 },
        { type: "line", label: "Fit", data: fitPoints, borderColor: "crimson", borderWidth: 2, pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { type: "linear", title: { display: true, text: `${BENCHMARK} excess return (%)` } },
        y: { title: { display: true, text: `${tk} excess return (%)` } },
      },
    },
  });
}

function renderBiasChart(canvas, biasDict, window, asset_tks) {
  const key = window === 5 ? "bias5" : "bias20";
  const hi = window === 5 ? 3.0 : 5.0;
  const datasets = asset_tks.map((tk, i) => ({
    label: tk,
    data: biasDict[tk].dates.map((d, j) => ({ x: ymd(d), y: biasDict[tk][key][j] })).filter(p => p.y != null),
    borderColor: COLORS[i % 4], borderWidth: 2, pointRadius: 0, fill: false,
  }));
  const refDates = biasDict[asset_tks[0]].dates.map(ymd);
  datasets.push({ label: `Overbought +${hi}%`, data: refDates.map(d => ({ x: d, y: hi })), borderColor: "#ef4444", borderDash: [4, 4], borderWidth: 1, pointRadius: 0, fill: false });
  datasets.push({ label: `Oversold -${hi}%`, data: refDates.map(d => ({ x: d, y: -hi })), borderColor: "#22c55e", borderDash: [4, 4], borderWidth: 1, pointRadius: 0, fill: false });
  datasets.push({ label: "0", data: refDates.map(d => ({ x: d, y: 0 })), borderColor: "#aaaaaa", borderDash: [2, 2], borderWidth: 1, pointRadius: 0, fill: false });
  makeChart(canvas, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      plugins: { title: { display: true, text: `MA${window} BIAS 乖離率 (%)` }, legend: { position: "bottom" } },
      scales: { x: { type: "category", ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: "BIAS (%)" } } },
    },
  });
}

// ── Rendering helpers ────────────────────────────────────────────────────
function td(text, cls) {
  return `<td${cls ? ` class="${cls}"` : ""}>${text}</td>`;
}
function buildTable(headers, rows) {
  const thead = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r => `<tr>${r.join("")}</tr>`).join("")}</tbody>`;
  return thead + tbody;
}
function metricBox(label, value, delta, deltaClass) {
  return `<div class="metric-box">
    <div class="metric-label">${label}</div>
    <div class="metric-value">${value}</div>
    ${delta ? `<div class="metric-delta ${deltaClass}">${delta}</div>` : ""}
  </div>`;
}
function downloadCSV(filename, headers, rows) {
  const lines = [headers.join(",")];
  for (const r of rows) lines.push(r.join(","));
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Main run ─────────────────────────────────────────────────────────────
function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = cls || "";
}

const DAY_MAP = { short: 365, medium: Math.round(365 * 1.75), long: 365 * 3 };

async function run() {
  const runBtn = document.getElementById("runBtn");
  runBtn.disabled = true;
  document.getElementById("results").classList.add("hidden");
  destroyCharts();

  const tickers = [];
  for (let i = 0; i < 4; i++) {
    const v = document.getElementById(`tk${i}`).value.trim().toUpperCase();
    if (v) tickers.push(v);
  }
  if (tickers.length === 0) {
    setStatus("Enter at least one ticker.", "error");
    runBtn.disabled = false;
    return;
  }

  const tfKey = document.getElementById("timeframe").value;
  const freq = document.getElementById("frequency").value;
  const today = dateOnly(new Date());
  const startDate = dateOnly(addDays(today, -DAY_MAP[tfKey]));
  const period1 = startDate;
  const period2 = addDays(today, 1);

  let rfAnnual, rfLabel;
  const rfOption = document.getElementById("rfOption").value;
  if (rfOption === "fixed") {
    const rfPct = Number(document.getElementById("rfPct").value);
    rfAnnual = rfPct / 100;
    rfLabel = `Fixed ${rfPct.toFixed(1)}%`;
  } else {
    setStatus("Fetching 0050 dividend yield…");
    const { yieldRate, totalDiv, price } = await fetch0050Yield();
    rfAnnual = yieldRate;
    rfLabel = `0050 Yield ${(yieldRate * 100).toFixed(2)}%`;
    if (price) {
      setStatus(`0050: 12-month Dividends NT$${totalDiv.toFixed(2)} | Price NT$${price.toFixed(2)} → Yield ${(yieldRate * 100).toFixed(2)}%`, "success");
    }
  }

  const allTickers = [...new Set([...tickers, BENCHMARK])];
  setStatus(`Downloading data for ${allTickers.join(", ")} …`);

  const seriesMap = {};
  const failed = [];
  await Promise.all(allTickers.map(async tk => {
    try {
      const result = await fetchYahoo(tk, period1, period2, "1d");
      seriesMap[tk] = parseSeries(result);
    } catch (e) {
      failed.push(tk);
    }
  }));

  if (failed.length) {
    setStatus(`Failed to fetch: ${failed.join(", ")} — excluded.`, "error");
  }
  if (!seriesMap[BENCHMARK]) {
    setStatus(`Benchmark ${BENCHMARK} unavailable. Check your connection.`, "error");
    runBtn.disabled = false;
    return;
  }

  const assetTks = tickers.filter(tk => seriesMap[tk] && tk !== BENCHMARK);
  if (assetTks.length === 0) {
    setStatus("No valid asset tickers found.", "error");
    runBtn.disabled = false;
    return;
  }

  // ── Align adjusted-close price data ──
  const alignedMap = {};
  for (const tk of [...assetTks, BENCHMARK]) alignedMap[tk] = seriesMap[tk];
  const { dates: priceDates, values: priceAligned } = alignSeries(alignedMap);
  if (priceDates.length < 10) {
    setStatus("Insufficient aligned data — check ticker symbols.", "error");
    runBtn.disabled = false;
    return;
  }

  // ── Returns ──
  let retDates, retAligned, fMult;
  if (freq === "W") {
    const idx = weeklyIndices(priceDates);
    retDates = idx.slice(1).map(i => priceDates[i]);
    retAligned = {};
    for (const tk of [...assetTks, BENCHMARK]) {
      const weeklyVals = idx.map(i => priceAligned[tk][i]);
      retAligned[tk] = pctChange(weeklyVals);
    }
    fMult = 52;
  } else {
    retDates = priceDates.slice(1);
    retAligned = {};
    for (const tk of [...assetTks, BENCHMARK]) retAligned[tk] = pctChange(priceAligned[tk]);
    fMult = 252;
  }

  const rf = periodRf(rfAnnual, freq);
  const excBenchAll = retAligned[BENCHMARK].map(r => r - rf);

  const results = {};
  for (const tk of assetTks) {
    const excAsset = retAligned[tk].map(r => r - rf);
    const reg = linregress(excBenchAll, excAsset);
    results[tk] = {
      beta: reg.slope,
      alphaPeriod: reg.intercept,
      r2: reg.r2,
      se: reg.se,
      pvalue: reg.pvalue,
      excAsset, excBench: excBenchAll,
      alphaAnnualPct: annualiseAlpha(reg.intercept, freq),
    };
  }

  // ── BIAS (raw close, optionally extended with live price) ──
  setStatus("Computing BIAS …");
  const bias_dict = {};
  let biasIsLive = false;
  await Promise.all(assetTks.map(async tk => {
    const s = seriesMap[tk];
    let dates = s.dates.slice();
    let close = s.close.slice();
    const live = await fetchLivePrice(tk);
    if (live != null) {
      const todayStr = ymd(today);
      if (ymd(dates[dates.length - 1]) === todayStr) {
        close[close.length - 1] = live;
      } else {
        dates = [...dates, today];
        close = [...close, live];
      }
      biasIsLive = true;
    }
    bias_dict[tk] = computeBias(dates, close);
  }));

  // ── Render ──
  setStatus("");
  document.getElementById("results").classList.remove("hidden");

  const cfg = { tickers, assetTks, freq, freqLabel: freq === "D" ? "Daily" : "Weekly", rfAnnual, rfLabel, fMult, timeframe: tfKey };
  renderTab1(priceDates, priceAligned, results, assetTks, cfg, bias_dict, biasIsLive);
  renderTab2(results, assetTks, cfg);
  renderTab3(priceDates, priceAligned, retDates, retAligned, results, assetTks, cfg);

  runBtn.disabled = false;
}

const TIMEFRAME_LABELS = { short: "Short Term (≤ 1Y)", medium: "Medium Term (1 – 2.5Y)", long: "Long Term (> 2.5Y)" };

function renderTab1(priceDates, priceAligned, results, assetTks, cfg, bias_dict, biasIsLive) {
  // Context bar
  const latestDate = priceDates[priceDates.length - 1];
  const lagDays = Math.round((dateOnly(new Date()) - new Date(latestDate + "T00:00:00")) / 86400000);
  const dateBoxClass = lagDays > 2 ? "ctx-box warn" : "ctx-box";
  const dateLabel = lagDays > 2 ? `Data as of: ${latestDate} ⚠️ (${lagDays}d lag)` : `Data as of: ${latestDate}`;
  document.getElementById("contextBar").innerHTML = `
    <div class="ctx-box">Period: <strong>${TIMEFRAME_LABELS[cfg.timeframe]}</strong></div>
    <div class="ctx-box">Frequency: <strong>${cfg.freqLabel}</strong></div>
    <div class="ctx-box">Rf: <strong>${cfg.rfLabel}</strong></div>
    <div class="${dateBoxClass}">${dateLabel}</div>
  `;

  // KPI table
  const kpiRows = assetTks.map(tk => {
    const res = results[tk];
    const a = res.alphaAnnualPct, b = res.beta, r2 = res.r2;
    const signal = a >= 0 ? "✅ Outperforms" : "❌ Underperforms";
    const vol = Math.abs(b) > 1 ? "Aggressive" : "Defensive";
    return [
      td(tk), td(`${a >= 0 ? "+" : ""}${a.toFixed(2)}%`, cellClass(`${a >= 0 ? "+" : ""}${a.toFixed(2)}%`)),
      td(b.toFixed(4)), td(r2.toFixed(4)), td(signal, cellClass(signal)), td(vol, cellClass(vol)),
    ];
  });
  document.getElementById("kpiTable").innerHTML = buildTable(
    ["Ticker", "Ann. Alpha", "Beta (β)", "R²", "Signal", "Volatility"], kpiRows
  );

  // Per-ticker detail
  const detailHtml = assetTks.map(tk => {
    const res = results[tk];
    const a = res.alphaAnnualPct, b = res.beta, r2 = res.r2;
    return `<details>
      <summary>${tk} | α ${a >= 0 ? "+" : ""}${a.toFixed(2)}% · β ${b.toFixed(4)} · R² ${r2.toFixed(4)}</summary>
      <div class="metrics-row">
        ${metricBox("Ann. Alpha (α)", `${a >= 0 ? "+" : ""}${a.toFixed(2)}%`, a >= 0 ? "Outperforms" : "Underperforms", a >= 0 ? "pos" : "neg")}
        ${metricBox("Beta (β)", b.toFixed(4), Math.abs(b) > 1 ? "Aggressive" : "Defensive", "neutral")}
        ${metricBox("R²", r2.toFixed(4), `${(r2 * 100).toFixed(1)}% from ${BENCHMARK}`, "neutral")}
      </div>
    </details>`;
  }).join("");
  document.getElementById("tickerDetail").innerHTML = detailHtml;

  // BIAS table
  const biasRows = [];
  for (const tk of assetTks) {
    const d = bias_dict[tk];
    let lastIdx = d.bias5.length - 1;
    while (lastIdx >= 0 && (d.bias5[lastIdx] == null || d.bias20[lastIdx] == null)) lastIdx--;
    if (lastIdx < 0) continue;
    const b5 = d.bias5[lastIdx], b20 = d.bias20[lastIdx];
    const sig5 = biasSignal(b5, 5), sig20 = biasSignal(b20, 20);
    biasRows.push([
      td(tk),
      td(`${b5 >= 0 ? "+" : ""}${b5.toFixed(2)}%`, cellClass(`${b5 >= 0 ? "+" : ""}${b5.toFixed(2)}%`)),
      td(sig5, cellClass(sig5)),
      td(`${b20 >= 0 ? "+" : ""}${b20.toFixed(2)}%`, cellClass(`${b20 >= 0 ? "+" : ""}${b20.toFixed(2)}%`)),
      td(sig20, cellClass(sig20)),
    ]);
  }
  const biasTableEl = document.getElementById("biasTable");
  const biasChartDetails = document.getElementById("biasChartDetails");
  if (biasRows.length) {
    biasTableEl.innerHTML = buildTable(["Ticker", "Bias_5 (%)", "5D Signal", "Bias_20 (%)", "20D Signal"], biasRows);
    let liveTag;
    if (biasIsLive) {
      if (isMarketOpen()) liveTag = "🔴 LIVE";
      else liveTag = `📊 Today's Close ${ymd(new Date()).slice(5)}`;
    } else {
      liveTag = `📊 EOD — Data as of ${priceDates[priceDates.length - 1].slice(5)}`;
    }
    document.getElementById("biasCaption").textContent = `${liveTag} | Thresholds — Bias_5: ±3% | Bias_20: ±5%`;
    biasChartDetails.classList.remove("hidden");
    renderBiasChart(document.getElementById("bias5Canvas"), bias_dict, 5, assetTks);
    renderBiasChart(document.getElementById("bias20Canvas"), bias_dict, 20, assetTks);
  } else {
    biasTableEl.innerHTML = "";
    document.getElementById("biasCaption").textContent = "";
    biasChartDetails.classList.add("hidden");
  }

  // Cumulative returns chart
  renderCumulativeChart(document.getElementById("cumCanvas"), priceDates, priceAligned, [...assetTks, BENCHMARK]);
}

function renderTab2(results, assetTks, cfg) {
  document.getElementById("scatterCaption").textContent =
    `X-axis: ${BENCHMARK} excess return | Y-axis: Asset excess return | Regression line shows CAPM fit | Rf = ${cfg.rfLabel}`;

  const grid = document.getElementById("scatterGrid");
  grid.innerHTML = assetTks.map((tk, i) => `
    <div class="chart-box">
      <div class="chart-title">${tk}</div>
      <canvas id="scatter-${i}"></canvas>
    </div>
  `).join("");
  assetTks.forEach((tk, i) => renderScatterChart(document.getElementById(`scatter-${i}`), tk, results[tk], COLORS[i % 4]));

  // Regression summary table
  const regRows = assetTks.map(tk => {
    const r = results[tk];
    return [
      td(tk), td(`${(r.alphaPeriod * 100).toFixed(5)}%`), td(`${r.alphaAnnualPct >= 0 ? "+" : ""}${r.alphaAnnualPct.toFixed(2)}%`),
      td(r.beta.toFixed(4)), td(r.r2.toFixed(4)), td(r.pvalue.toFixed(6)), td(r.se.toFixed(7)), td(r.excAsset.length),
    ];
  });
  document.getElementById("regTable").innerHTML = buildTable(
    ["Ticker", "α (per period)", "α (annualised)", "β (Beta)", "R²", "P-value", "Std Error", "N obs"], regRows
  );

  // Per-asset detail
  document.getElementById("perAssetDetail").innerHTML = assetTks.map((tk, i) => {
    const r = results[tk];
    const a = r.alphaAnnualPct, b = r.beta, r2 = r.r2;
    let bTxt;
    if (b > 1.2) bTxt = `🔴 Aggressive — ${b.toFixed(2)}× market sensitivity`;
    else if (b > 0.8) bTxt = `🟡 Market-like — ${b.toFixed(2)}× market sensitivity`;
    else if (b >= 0) bTxt = `🟢 Defensive — ${b.toFixed(2)}× market sensitivity`;
    else bTxt = `🔵 Inverse — ${b.toFixed(2)}× market sensitivity`;
    const aTxt = a >= 0 ? `✅ Outperforms by ${a.toFixed(2)}%/yr` : `❌ Underperforms by ${Math.abs(a).toFixed(2)}%/yr`;
    const r2Cat = r2 > 0.7 ? "Strong" : r2 > 0.4 ? "Moderate" : "Weak";
    const r2Txt = `${r2Cat} — ${(r2 * 100).toFixed(1)}% of variance from ${BENCHMARK}`;
    return `<details${i === 0 ? " open" : ""}>
      <summary>🔍 ${tk} | α = ${a >= 0 ? "+" : ""}${a.toFixed(2)}% β = ${b.toFixed(4)} R² = ${r2.toFixed(4)}</summary>
      <div class="interp-row">
        <div><canvas id="scatter-detail-${i}"></canvas></div>
        <div>
          <table>
            <tr><td>α per period</td><td>${(r.alphaPeriod * 100).toFixed(5)}%</td></tr>
            <tr><td>Annualised α</td><td>${a >= 0 ? "+" : ""}${a.toFixed(2)}%</td></tr>
            <tr><td>β (Beta)</td><td>${b.toFixed(4)}</td></tr>
            <tr><td>R²</td><td>${r2.toFixed(4)}</td></tr>
            <tr><td>P-value</td><td>${r.pvalue.toFixed(6)}</td></tr>
            <tr><td>Std Error</td><td>${r.se.toFixed(7)}</td></tr>
            <tr><td>Observations</td><td>${r.excAsset.length}</td></tr>
          </table>
          <hr>
          <p><strong>β:</strong> ${bTxt}</p>
          <p><strong>α:</strong> ${aTxt}</p>
          <p><strong>R²:</strong> ${r2Txt}</p>
        </div>
      </div>
    </details>`;
  }).join("");
  assetTks.forEach((tk, i) => renderScatterChart(document.getElementById(`scatter-detail-${i}`), tk, results[tk], COLORS[i % 4]));
}

function renderTab3(priceDates, priceAligned, retDates, retAligned, results, assetTks, cfg) {
  const allTks = [...assetTks, BENCHMARK];

  // Price table
  document.getElementById("returnsHeader").textContent = `${cfg.freqLabel} Returns (%)`;
  const priceRows = priceDates.map((d, i) => [td(d), ...allTks.map(tk => td(priceAligned[tk][i].toFixed(2)))]);
  document.getElementById("priceTable").innerHTML = buildTable(["Date", ...allTks], priceRows);

  const returnsRows = retDates.map((d, i) => [td(d), ...allTks.map(tk => td((retAligned[tk][i] * 100).toFixed(4)))]);
  document.getElementById("returnsTable").innerHTML = buildTable(["Date", ...allTks], returnsRows);

  // Descriptive statistics
  const statRows = allTks.map(tk => {
    const s = retAligned[tk];
    const annRet = (Math.pow(1 + mean(s), cfg.fMult) - 1) * 100;
    const annVol = std(s) * Math.sqrt(cfg.fMult) * 100;
    const sharpe = annVol ? (annRet - cfg.rfAnnual * 100) / annVol : 0;
    const mdd = maxDrawdownFromReturns(s);
    return [
      td(tk), td(`${annRet.toFixed(2)}%`), td(`${annVol.toFixed(2)}%`), td(sharpe.toFixed(3)),
      td(`${mdd.toFixed(2)}%`), td(skewness(s).toFixed(3)), td(kurtosisExcess(s).toFixed(3)), td(s.length),
    ];
  });
  document.getElementById("statsTable").innerHTML = buildTable(
    ["Ticker", "Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Max Drawdown", "Skewness", "Kurtosis", "N obs"], statRows
  );

  // Correlation matrix
  const corrHeader = ["", ...allTks];
  const corrRows = allTks.map(tk1 => {
    const cells = allTks.map(tk2 => {
      const c = correlation(retAligned[tk1], retAligned[tk2]);
      return `<td style="background-color:${corrColor(c)}">${c.toFixed(3)}</td>`;
    });
    return [`<td><strong>${tk1}</strong></td>`, ...cells];
  });
  document.getElementById("corrTable").innerHTML = buildTable(corrHeader, corrRows);

  // CSV export
  document.getElementById("dlPrices").onclick = () => {
    downloadCSV("prices.csv", ["Date", ...allTks], priceDates.map((d, i) => [d, ...allTks.map(tk => priceAligned[tk][i].toFixed(4))]));
  };
  document.getElementById("dlReturns").onclick = () => {
    downloadCSV("returns.csv", ["Date", ...allTks], retDates.map((d, i) => [d, ...allTks.map(tk => (retAligned[tk][i] * 100).toFixed(6))]));
  };
}

// ── Init ─────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("runBtn").addEventListener("click", run);

  document.getElementById("timeframe").addEventListener("change", e => {
    const freqEl = document.getElementById("frequency");
    freqEl.value = e.target.value === "short" ? "D" : "W";
  });

  document.getElementById("rfOption").addEventListener("change", e => {
    document.getElementById("rfFixedRow").classList.toggle("hidden", e.target.value !== "fixed");
  });

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  document.querySelectorAll(".chart-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".chart-tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".chart-box-toggle").forEach(c => c.classList.add("hidden-chart"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.chart).classList.remove("hidden-chart");
    });
  });
});
