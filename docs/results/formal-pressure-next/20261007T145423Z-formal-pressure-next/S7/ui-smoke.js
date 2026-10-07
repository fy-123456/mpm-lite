const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync(process.argv[2],'utf8');const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
let arcs=0;const ctx={clearRect(){},fillRect(){},strokeRect(){},beginPath(){},fill(){},arc(x,y,r){if(![x,y,r].every(Number.isFinite))throw Error('nonfinite drawing');arcs++;}};
const elements={};for(const id of ['view','frame','scale','color','status','play'])elements[id]={value:id==='scale'?'1000':id==='color'?'p':'0',addEventListener(e,f){this.fn=f;},getContext(){return ctx;}};
let tick;const box={document:{getElementById(id){return elements[id];}},setInterval(f){tick=f;},console};vm.createContext(box);vm.runInContext(script,box);
for(const n of [0,2,4])for(const mode of ['p','u','J'])for(const amp of [1,1000,10000]){elements.frame.value=String(n);elements.color.value=mode;elements.scale.value=String(amp);vm.runInContext('draw()',box);}
elements.play.onclick();tick();if(!elements.status.textContent.includes('帧'))throw Error('missing status');
console.log(JSON.stringify({passed:true,arcs,frames:[0,2,4],modes:['p','u','J'],scales:[1,1000,10000],playback:true,real_browser:false}));
