const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');
const queryInput = document.getElementById('queryInput');
const answerArea = document.getElementById('answerArea');
const status = document.getElementById('status');

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
  }catch(e){
    setStatus('Network error');
    answerArea.textContent = String(e);
  }
}

askBtn.addEventListener('click', ask);
clearBtn.addEventListener('click', ()=>{queryInput.value=''; answerArea.textContent=''; setStatus('Ready')});
queryInput.addEventListener('keydown', (e)=>{if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ask()}})
