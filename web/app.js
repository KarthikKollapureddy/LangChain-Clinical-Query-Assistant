const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');
const queryInput = document.getElementById('queryInput');
const answerArea = document.getElementById('answerArea');
const status = document.getElementById('status');
const sourcesList = document.getElementById('sourcesList');

function setStatus(s){status.textContent = s}

async function ask(){
  const q = queryInput.value.trim();
  if(!q) return;
  setStatus('Thinking...');
  answerArea.textContent = '';
  try{
    const resp = await fetch('/query',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})
    });
    if(!resp.ok){
      const err = await resp.json().catch(()=>({detail:'Unknown error'}));
      setStatus('Error');
      answerArea.textContent = JSON.stringify(err, null, 2);
      return;
    }
    const data = await resp.json();
    setStatus('Done');
    answerArea.textContent = data.answer || JSON.stringify(data, null, 2);
    // render sources
    sourcesList.innerHTML = '';
    if(Array.isArray(data.sources)){
      data.sources.forEach(s=>{
        const li = document.createElement('li');
        const meta = document.createElement('div'); meta.className='src-meta'; meta.textContent = s.source || s.id || 'source';
        const txt = document.createElement('div'); txt.className='src-text'; txt.textContent = s.text || '';
        const actions = document.createElement('div'); actions.className='src-actions';
        const copyBtn = document.createElement('button'); copyBtn.textContent='Copy';
        copyBtn.onclick = ()=>{navigator.clipboard.writeText(s.text || '');};
        actions.appendChild(copyBtn);
        li.appendChild(meta); li.appendChild(txt); li.appendChild(actions);
        sourcesList.appendChild(li);
      })
    }
  }catch(e){
    setStatus('Network error');
    answerArea.textContent = String(e);
  }
}

askBtn.addEventListener('click', ask);
clearBtn.addEventListener('click', ()=>{queryInput.value=''; answerArea.textContent=''; setStatus('Ready')});
queryInput.addEventListener('keydown', (e)=>{if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ask()}})
