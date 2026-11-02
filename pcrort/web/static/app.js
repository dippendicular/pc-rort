"use strict";
let csrf = "";
const el = id => document.getElementById(id);
function notice(message, error=false) { el("notice").textContent = message; el("notice").classList.toggle("error",error); }
async function call(path, options={}) {
  options.credentials="same-origin";
  if(options.method && options.method!=="GET" && csrf) options.headers={...(options.headers||{}),"X-CSRF-Token":csrf};
  const res=await fetch(path,options);
  const body=await res.json().catch(()=>({detail:"Unexpected response"}));
  if(!res.ok) throw new Error(body.detail||"Request failed");
  return body;
}
function makeItem(title,detail,status) {
  const row=document.createElement("div");row.className="item";
  const col=document.createElement("div");const strong=document.createElement("strong");
  strong.textContent=title;const small=document.createElement("small");small.textContent=detail;
  col.append(strong,small);row.append(col);
  if(status){const badge=document.createElement("span");badge.className="badge";badge.textContent=status;row.append(badge);}
  return row;
}
function populate(id,rows,message,renderer) {
  const root=el(id);root.replaceChildren();
  if(!rows.length){root.textContent=message;return;}
  rows.forEach(row=>root.append(renderer(row)));
}
async function refresh() {
  try {
    const entries=await Promise.all([call("/datasets"),call("/jobs"),call("/models")]);
    const datasets=entries[0],jobs=entries[1],models=entries[2];
    el("dataset-count").textContent=datasets.length;
    el("job-count").textContent=jobs.length;
    el("model-count").textContent=models.length;
    populate("datasets",datasets,"No datasets uploaded.",d=>makeItem(d.name||d.id,"ID: "+d.id,d.status));
    const training=jobs.filter(j=>j.model_mode!=="tabular"),runs=jobs.filter(j=>j.model_mode==="tabular");
    populate("jobs",training,"No training jobs yet.",j=>makeItem(j.model_mode||"Job","ID: "+j.id,j.status));
    populate("tabular-runs",runs,"No runs yet.",j=>{
      const row=makeItem("Evaluation","ID: "+j.id,j.status);
      if(j.status==="COMPLETED"){
        const link=document.createElement("a");link.href="/tabular/runs/"+j.id+"/download";link.textContent="Download results";
        const show=document.createElement("button");show.className="link";show.textContent="Summary";
        show.addEventListener("click",()=>action(async()=>{el("tabular-result").textContent=JSON.stringify(await call("/tabular/runs/"+j.id+"/results"),null,2);}));
        row.append(link,show);
      }
      return row;
    });
    const tabularSel=el("tabular-datasets");tabularSel.replaceChildren();
    datasets.filter(d=>d.kind==="tabular"&&d.status==="READY").forEach(d=>tabularSel.add(new Option(d.name+" ("+d.manifest.rows+" rows, target "+d.manifest.target+")",d.id)));
    populate("model-list",models,"No models available.",m=>makeItem(m.mode||"Model","ID: "+m.id,m.published?"Published":"Private"));
    const datasetSel=el("job-dataset");datasetSel.replaceChildren(new Option("Select a dataset",""));
    datasets.filter(d=>d.status==="READY").forEach(d=>datasetSel.add(new Option(d.name||d.id,d.id)));
    const modelSel=el("model-id");modelSel.replaceChildren(new Option("Select a model",""));
    models.filter(m=>m.mode==="radiology").forEach(m=>modelSel.add(new Option(m.id,m.id)));
    el("logout").hidden=false; notice("Workspace synchronized.");
  } catch(e){notice(e.message,true);}
}
function action(fn){Promise.resolve().then(fn).catch(e=>notice(e.message,true));}
el("auth").addEventListener("submit",e=>{
  e.preventDefault();const mode=e.submitter?.dataset.mode||"login";
  action(async()=>{
    const r=await call("/auth/"+mode,{method:"POST",body:new FormData(e.currentTarget)});
    if(mode==="register"){notice("Account created. Sign in to continue.");return;}
    csrf=r.csrf;await refresh();
  });
});
el("logout").addEventListener("click",()=>action(async()=>{await call("/auth/logout",{method:"POST"});location.reload();}));
el("refresh").addEventListener("click",()=>action(refresh));
el("upload-form").addEventListener("submit",e=>{
  e.preventDefault();action(async()=>{
    const f=new FormData();f.append("archive",el("dataset").files[0]);
    for(const [key,id] of [["target","upload-target"],["positive","upload-positive"],["exclude","upload-exclude"]]){
      if(el(id).value.trim()) f.append(key,el(id).value.trim());
    }
    const r=await call("/datasets",{method:"POST",body:f});
    const m=r.manifest||{};
    notice(m.kind==="tabular"?`Dataset ready: ${m.rows} rows, target "${m.target}", positive "${m.positive_label}"`:"Dataset uploaded: "+r.dataset_id);
    await refresh();
  });
});
el("job-form").addEventListener("submit",e=>{
  e.preventDefault();action(async()=>{
    const f=new FormData();f.append("dataset_id",el("job-dataset").value);f.append("model_mode",el("job-mode").value);
    const r=await call("/jobs",{method:"POST",body:f});
    notice("Queued training job: "+r.job_id);await refresh();
  });
});
el("tabular-form").addEventListener("submit",e=>{
  e.preventDefault();action(async()=>{
    const dataset_ids=[...el("tabular-datasets").selectedOptions].map(o=>o.value);
    const r=await call("/tabular/runs",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({dataset_ids,repetitions:Number(el("tabular-reps").value),fedavg_rounds:Number(el("tabular-rounds").value)})});
    notice("Evaluation queued: "+r.job_id);await refresh();
  });
});
el("predict-form").addEventListener("submit",e=>{
  e.preventDefault();action(async()=>{
    const f=new FormData();f.append("model_id",el("model-id").value);f.append("image",el("image").files[0]);
    const r=await call("/predictions",{method:"POST",body:f});
    el("result").textContent=JSON.stringify(r,null,2);notice("Inference complete.");
  });
});
fetch("/api/health").then(r=>r.json()).then(r=>el("health").textContent="API "+r.status).catch(()=>el("health").textContent="API unavailable");
call("/auth/session").then(r=>{csrf=r.csrf;return refresh();}).catch(()=>{});

el("transport-form").addEventListener("submit",e=>{
  e.preventDefault();action(async()=>{
    const center=JSON.parse(el("matrix-input").value);
    const rho=Number(el("matrix-rho").value);
    const radius=Number(el("matrix-radius").value);
    const r=await call("/api/transport/optimize",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({center,rho,radius,max_iterations:30})
    });
    el("transport-result").textContent=JSON.stringify(r,null,2);
    notice("Optimization completed.");
  });
});
