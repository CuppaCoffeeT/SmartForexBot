// Track current selection (last symbol shown on the chart)
let currentSymbol = null;
let currentInterval = '1h';
let activeTab = 'chat';
let overviewData = [];
let overviewSelected = null;
let overviewLoading = false;
let currentFeatureMode = 'fib';

async function postJSON(url, payload){
  const res = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!res.ok) throw new Error('Request failed');
  return res.json();
}

function messageMentionsPair(text){
  if(!text) return false;
  return /\b([A-Za-z]{3})[\/\s-]?([A-Za-z]{3})\b/.test(text);
}

function isBestTradeQuery(text){
  if(!text) return false;
  const t = text.toLowerCase();
  if(/best\s+pair|best\s+trade|what'?s\s+the\s+best/.test(t)) return true;
  if(/\b(best|top)\b/.test(t) && /trade|pair|setup|signal/.test(t)) return true;
  if(/recommend/.test(t) && !messageMentionsPair(text)) return true;
  return false;
}

function isFollowupQuery(text){
  if(!text) return false;
  if(messageMentionsPair(text)) return false;
  const t = text.toLowerCase();
  if(isBestTradeQuery(text)) return false;
  if(/\b(update|status|happening|why|explain|confidence|risk|chart|show|what'?s\s+going|tell\s+me\s+more)\b/.test(t)) return true;
  if(/^\s*(what'?s\s+next|and\s+now\?|now\s+what)\s*$/.test(t)) return true;
  return false;
}

function addMsg(text, who){
  const d=document.createElement('div');
  d.className='msg '+who;
  d.textContent=text;
  const log = document.querySelector('#chatlog');
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

function plotChart(ohlc, levels, title, entry, direction, targetId = 'chart'){
  if(!ohlc) return;
  const times = (ohlc.time || []).map(t=>new Date(t));
  if(!times.length) return;
  const trace = {x:times, open:ohlc.Open, high:ohlc.High, low:ohlc.Low, close:ohlc.Close, type:'candlestick', name:'OHLC'};
  const shapes = [];
  const annotations = [];
  const fibNames = Object.keys(levels||{});
  fibNames.forEach(k=>{
    const y = levels[k];
    shapes.push({type:'line',xref:'x',yref:'y',x0:times[0],x1:times[times.length-1],y0:y,y1:y,line:{dash:'dot',color:'#888'}});
    annotations.push({
      xref:'paper', x:0.995, xanchor:'right',
      yref:'y', y, showarrow:false,
      text:String(k), align:'right',
      font:{size:10,color:'#555'},
      bgcolor:'rgba(255,255,255,0.6)'
    });
  });

  const entryZone = (() => {
    if(!entry) return null;
    const lo = Number(entry.zone_low);
    const hi = Number(entry.zone_high);
    if(Number.isFinite(lo) && Number.isFinite(hi) && hi > lo){
      return {lo, hi, label: entry.fib_label || null};
    }
    return null;
  })();

  if(entryZone){
    const bandColor = direction === 'SELL'
      ? 'rgba(220, 20, 60, 0.12)'
      : direction === 'BUY'
        ? 'rgba(34, 139, 34, 0.12)'
        : 'rgba(30, 144, 255, 0.12)';
    shapes.push({
      type:'rect',
      xref:'paper', x0:0, x1:1,
      yref:'y', y0:entryZone.lo, y1:entryZone.hi,
      fillcolor:bandColor,
      line:{width:0}
    });
    annotations.push({
      xref:'paper', x:0.02, xanchor:'left',
      yref:'y', y:(entryZone.lo + entryZone.hi) / 2,
      showarrow:false,
      text: entryZone.label ? `Entry zone (${entryZone.label})` : 'Entry zone',
      font:{size:11,color:'#333'},
      bgcolor:'rgba(255,255,255,0.65)',
      bordercolor:'rgba(0,0,0,0.15)', borderwidth:1
    });
  }

  const layout = {
    title,
    dragmode:'pan',
    xaxis:{rangeslider:{visible:false}},
    yaxis:{fixedrange:false},
    shapes,
    annotations
  };
  Plotly.newPlot(targetId,[trace], layout, {responsive:true});
}

function prettyLabel(sym){
  try{
    const clean = String(sym).replace(/=X$/,'');
    if(clean.length === 6) return clean.slice(0,3) + '/' + clean.slice(3);
    return clean;
  }catch{ return sym; }
}

function shortState(state){
  return ({
    in_zone:'In zone',
    climbing_toward:'Climbing',
    dropping_toward:'Dropping',
    leaving_above:'Leaving ↑',
    leaving_below:'Leaving ↓',
    no_event:'Idle',
    unavailable:'—'
  })[state] || '—';
}

function colorForState(item){
  switch(item?.state){
    case 'in_zone': return '#ffe29a';
    case 'climbing_toward': return '#c6f6d5';
    case 'dropping_toward': return '#fed7d7';
    case 'leaving_above':
    case 'leaving_below': return '#e2e8f0';
    case 'no_event': return '#f7fafc';
    default: return '#f1f5f9';
  }
}

function formatPrice(v){
  return Number.isFinite(v) ? Number(v).toFixed(5) : '—';
}

function markOverviewSelection(){
  const cards = document.querySelectorAll('.overview-item');
  cards.forEach(card => {
    card.classList.toggle('selected', card.dataset.symbol === overviewSelected);
  });
}

function renderOverviewGrid(){
  const grid = document.getElementById('overviewGrid');
  if(!grid) return;
  grid.innerHTML = '';
  if(!overviewData.length){
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No retracement setups right now.';
    grid.appendChild(empty);
    return;
  }
  overviewData.forEach(item => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'overview-item';
    btn.dataset.symbol = item.symbol;
    btn.style.setProperty('--state-color', colorForState(item));
    btn.title = item.state_text || '';
    btn.innerHTML = `
      <span class="symbol">${prettyLabel(item.symbol)}</span>
      <span class="direction">${item.direction || '—'}</span>
      <span class="state">${shortState(item.state)}</span>
    `;
    btn.addEventListener('click', () => selectOverviewPair(item.symbol));
    grid.appendChild(btn);
  });
  markOverviewSelection();
}

async function loadOverview(force = false){
  if(overviewLoading && !force) return;
  overviewLoading = true;
  const grid = document.getElementById('overviewGrid');
  if(grid){
    grid.innerHTML = '<div class="loading">Loading overview…</div>';
  }
  try{
    const interval = currentInterval || '1h';
    const res = await fetch(`/api/overview?interval=${encodeURIComponent(interval)}&feature_mode=${encodeURIComponent(currentFeatureMode)}`);
    if(!res.ok) throw new Error('overview request failed');
    const data = await res.json();
    overviewData = data.pairs || [];
    const prev = overviewSelected;
    const stillExists = overviewData.some(item => item.symbol === prev);
    overviewSelected = stillExists ? prev : (overviewData[0]?.symbol || null);
    renderOverviewGrid();
    if(overviewSelected){
      await selectOverviewPair(overviewSelected);
    }else{
      const header = document.getElementById('overviewDetailHeader');
      if(header) header.textContent = 'No retracement pairs available.';
      Plotly.purge('overviewChart');
      const meta = document.getElementById('overviewDetailMeta');
      if(meta) meta.innerHTML = '';
    }
  }catch(err){
    if(grid){ grid.innerHTML = '<div class="empty">Failed to load overview.</div>'; }
    const header = document.getElementById('overviewDetailHeader');
    if(header) header.textContent = 'Unable to load overview data.';
  }finally{
    overviewLoading = false;
  }
}

function buildDetailMeta(detail){
  const insight = detail?.insight || {};
  const entry = insight.entry || {};
  const rows = [
    ['Direction', insight.direction || '—'],
    ['Confidence', (Number.isFinite(insight.prob) && Number.isFinite(insight.threshold)) ? `${Number(insight.prob).toFixed(3)} vs ${Number(insight.threshold).toFixed(3)}` : '—'],
    ['Nearest Fib', entry.fib_label != null ? entry.fib_label : '—'],
    ['Entry zone', (Number.isFinite(entry.zone_low) && Number.isFinite(entry.zone_high)) ? `${formatPrice(entry.zone_low)} – ${formatPrice(entry.zone_high)}` : '—'],
    ['Last price', Number.isFinite(entry.close) ? formatPrice(entry.close) : '—'],
    ['Proximity', Number.isFinite(entry.proximity) ? `${entry.proximity.toFixed(2)}× zone` : '—']
  ];
  return rows.map(([label,value])=>`
    <div>
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join('');
}

async function selectOverviewPair(symbol){
  overviewSelected = symbol;
  markOverviewSelection();
  const header = document.getElementById('overviewDetailHeader');
  const meta = document.getElementById('overviewDetailMeta');
  if(header) header.textContent = 'Loading pair details…';
  if(meta) meta.innerHTML = '';
  try{
    const interval = currentInterval || '1h';
    const detail = await postJSON('/api/chat', {message:'', symbol, interval, followup:true, feature_mode: currentFeatureMode});
    const entry = detail?.insight?.entry;
    if(header) header.textContent = entry?.state_text || 'No retracement information available.';
    if(meta) meta.innerHTML = buildDetailMeta(detail);
    if(detail.ohlc){
      plotChart(detail.ohlc, detail.levels, `${symbol} (${interval}) — Candles & Fibs`, entry, detail.insight?.direction, 'overviewChart');
    }else{
      Plotly.purge('overviewChart');
    }
  }catch(err){
    if(header) header.textContent = 'Failed to load pair detail.';
    if(meta) meta.innerHTML = '';
  }
}

async function send(){
  const input = document.querySelector('#msg');
  const msg = input.value.trim();
  if(!msg) return;
  addMsg(msg,'user');
  input.value='';
  const interval = currentInterval || '1h';
  currentInterval = interval;
  try{
    const payload = {message: msg, interval, feature_mode: currentFeatureMode};
    if(currentSymbol && isFollowupQuery(msg)){
      payload.symbol = currentSymbol;
      payload.followup = true;
    }
    const data = await postJSON('/api/chat', payload);
    if(data.error){ addMsg('Error: '+data.error,'bot'); return; }
    addMsg(data.reply,'bot');
    if(data.ohlc){
      currentSymbol = data.symbol || currentSymbol;
      plotChart(data.ohlc, data.levels, (data.symbol||'')+' ('+interval+') — Candles & Fibs', data.insight?.entry, data.insight?.direction);
      const s = data.insight || {};
      const txt = s.has_event
        ? `Action: ${s.direction} | Score ${s.prob} vs thr ${s.threshold}`
        : 'No fresh Fib event.';
      document.getElementById('summary').textContent = txt;
    }
    if(data.recommendations){
      const box = document.getElementById('recos'); box.innerHTML='';
      data.recommendations.forEach(r=>{
        const d=document.createElement('div'); d.className='card';
        d.innerHTML = `<h4>${r.symbol}</h4><div>${r.direction}</div><small>conf: ${r.confidence} | score: ${r.prob}</small>`;
        box.appendChild(d);
      });
    }
  }catch(err){
    addMsg('Error: failed to reach the server.','bot');
  }
}

document.getElementById('send').addEventListener('click', send);
document.getElementById('msg').addEventListener('keydown', e=>{ if(e.key==='Enter') send(); });

document.getElementById('btnRefresh').addEventListener('click', ()=>{
  if(activeTab === 'overview'){
    loadOverview(true);
  }else{
    refreshChart(true);
  }
});

const featureSelect = document.getElementById('featureMode');
if(featureSelect){
  featureSelect.addEventListener('change', ()=>{
    currentFeatureMode = featureSelect.value || 'fib';
    if(activeTab === 'overview'){
      loadOverview(true);
    }else{
      // Clear current chart symbol to allow new best-trade selection if desired
      refreshChart(true);
    }
  });
}

function setTab(tab){
  if(tab === activeTab) return;
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(btn=>{
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  document.querySelectorAll('.view').forEach(view=>{
    view.classList.toggle('active', view.dataset.view === tab);
  });
  if(activeTab === 'overview'){
    loadOverview(true);
  }
}

document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click', ()=> setTab(btn.dataset.tab));
});

async function refreshChart(force = false){
  const interval = currentInterval || '1h';
  if(!currentSymbol){
    try{
      const j = await (await fetch(`/api/recommendations?feature_mode=${encodeURIComponent(currentFeatureMode)}`)).json();
      if(j.recommendations?.length){
        currentSymbol = j.recommendations[0].symbol;
      }
    }catch{}
  }
  if(!currentSymbol){ return; }
  try{
    const status = await postJSON('/api/chat', {message: '', symbol: currentSymbol, interval, followup:true, feature_mode: currentFeatureMode});
    if(status.ohlc){
      plotChart(status.ohlc, status.levels, currentSymbol+' ('+interval+') — Candles & Fibs', status.insight?.entry, status.insight?.direction);
      const s = status.insight || {};
      document.getElementById('summary').textContent = s.has_event
        ? `Action: ${s.direction} | Score ${s.prob} vs thr ${s.threshold}`
        : 'No fresh Fib event.';
    }
    const j = await (await fetch(`/api/recommendations?feature_mode=${encodeURIComponent(currentFeatureMode)}`)).json();
    const box = document.getElementById('recos'); box.innerHTML='';
    if(j.recommendations?.length){
      j.recommendations.forEach(r=>{
        const d=document.createElement('div'); d.className='card';
        d.innerHTML = `<h4>${r.symbol}</h4><div>${r.direction}</div><small>conf: ${r.confidence} | score: ${r.prob}</small>`;
        box.appendChild(d);
      });
    } else { box.textContent = 'No strong setups right now.'; }
  }catch(err){ /* ignore */ }
}
