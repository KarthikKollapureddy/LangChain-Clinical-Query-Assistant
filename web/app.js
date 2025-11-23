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
    // render retriever sources (these come from the indexed documents)
    sourcesList.innerHTML = '';
    if(Array.isArray(data.sources) && data.sources.length){
      // support two formats: array of strings (new) or array of objects (legacy)
      data.sources.forEach(s=>{
        const li = document.createElement('li');
        let metaText = '';
        let txtContent = '';
        if(typeof s === 'string'){
          metaText = s;
        } else if(typeof s === 'object' && s !== null){
          metaText = s.source || s.id || 'source';
          txtContent = s.text || '';
        } else {
          metaText = String(s);
        }
        const meta = document.createElement('div'); meta.className='src-meta'; meta.textContent = metaText;
        const txt = document.createElement('div'); txt.className='src-text'; txt.textContent = txtContent;
        const actions = document.createElement('div'); actions.className='src-actions';
        const copyBtn = document.createElement('button'); copyBtn.textContent='Copy';
        copyBtn.onclick = ()=>{navigator.clipboard.writeText(txtContent || metaText || '');};
        actions.appendChild(copyBtn);
        li.appendChild(meta); li.appendChild(txt); li.appendChild(actions);
        sourcesList.appendChild(li);
      })
    } else {
      const li = document.createElement('li'); li.textContent='No retriever sources found.'; sourcesList.appendChild(li);
    }

    // if the model reported its own sources, show them separately (collapsed)
    const modelSources = data.model_sources || [];
    const modelDiv = document.getElementById('modelSources');
    if(modelDiv){
      modelDiv.innerHTML = '';
      if(Array.isArray(modelSources) && modelSources.length){
        const hdr = document.createElement('div'); hdr.textContent='Model-cited sources (from answer):'; hdr.className='model-sources-hdr';
        modelDiv.appendChild(hdr);
        const ul = document.createElement('ul');
        modelSources.forEach(ms=>{const li=document.createElement('li'); li.textContent = ms; ul.appendChild(li)});
        modelDiv.appendChild(ul);
      } else {
        modelDiv.textContent = '';
      }
    }
  }catch(e){
    setStatus('Network error');
    answerArea.textContent = String(e);
  }
}

askBtn.addEventListener('click', ask);
clearBtn.addEventListener('click', ()=>{queryInput.value=''; answerArea.textContent=''; setStatus('Ready')});
queryInput.addEventListener('keydown', (e)=>{if(e.key==='Enter' && (e.ctrlKey||e.metaKey)){ask()}})
