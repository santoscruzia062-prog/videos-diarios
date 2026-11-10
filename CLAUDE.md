# Rotina diária de vídeos — manual de operação

Este repositório é a "memória" da rotina. Toda execução começa do zero: leia este arquivo
e o `historico.md` inteiros antes de criar qualquer coisa, e atualize o `historico.md` no final.

## O que a rotina faz (toda noite, ~21h, horário de Brasília)
1. Lê `historico.md` (não repetir formato da arena + mecanismo dos últimos 3 dias).
2. Escolhe **8 vídeos diferentes** para o DIA SEGUINTE no estilo **BOLINHAS COLORIDAS SE
   MULTIPLICANDO** (é o estilo que o dono quer agora — duelos/royale de times NÃO estão performando,
   não fazer mais). Cada vídeo combina um FORMATO de arena + um MECANISMO girando lá dentro.
3. Cria os vídeos com `motor/bolas.py`, confere quadros-chave (início, ~10 s, ~20 s, cheio,
   1 quadro antes da explosão, final) olhando as imagens, e só segue se estiverem bons.
4. Envia os vídeos para o branch `midia` (ver "Hospedagem") e agenda no Metricool:
   **07:00, 08:30, 10:00, 11:30 (manhã) e 13:30, 16:00, 18:30, 21:00 (tarde/noite)** do dia
   seguinte, nas 3 redes (pedido do dono: 8 por dia, 4 de manhã e 4 à tarde, das 7h às 21h).
   Intercale os tipos (não pôr dois parecidos seguidos).
5. Atualiza `historico.md` (branch `main`) e termina com um resumo curto.
   Não precisa de aprovação do dono: faça bem feito e agende direto.

## Padrão de qualidade (pedido do dono — obrigatório)
- Vertical 720×1280, 30 fps, ~40–45 s, H.264 + AAC, som próprio sintetizado (nunca música
  com direitos autorais, nunca logos/marcas).
- Bolinhas coloridas se multiplicam quando o mecanismo bate/esmaga. O mecanismo **gira sempre
  no mesmo sentido** (nada de vai-e-vem). Tique-taque o tempo todo + notas "tom, tom, trim, plim".
- Enche RÁPIDO e enche MUITO: na hora de explodir a arena tem que estar **LOTADA, sem buracos**
  (o dono reprovou um vídeo que explodiu com espaço vazio). Depois de cheio: **3 rodadas** com
  a borda balançando e tremendo cada vez mais, e só então explode com som forte.
- **Inovar sempre** o formato e o mecanismo. Exemplos pedidos pelo dono: quadrado com 4 bastões,
  triângulo com engrenagem, coração com 2 bastões, círculo com 4 bastões, hexágono com 3 bastões,
  retângulo com 2 bastões gigantes, círculo com 1 bolinha quicando (cada batida na borda = +1),
  relógio com ponteiro girando, quebra-quadrados (`quebra.py`). Ideias novas: estrela, losango, pentágono, cruz de bastões,
  bastões de tamanhos diferentes, engrenagens de 3 dentes, velocidades diferentes, paletas novas.
- **TEXTO DENTRO DO VÍDEO BILÍNGUE** (pedido do dono, 09/11): INGLÊS GRANDE em cima e
  PORTUGUÊS MENOR logo embaixo (cinza-claro). O motor já faz isso: em cada config ponha
  `hook` (inglês) + `hook_pt` (português); "Follow for more!"/"Segue a página pra ver mais",
  "ABOUT TO EXPLODE!"/"Vai explodir!", "N BALLS!"/"N bolinhas!", "Liked it? FOLLOW US!"/
  "Gostou? Segue a página!", "Share it with a friend"/"Compartilha com alguém".
  Motores novos devem seguir o mesmo padrão.
- **NÃO pedir para adivinhar** quantas bolinhas (o dono não quer). Gancho no 1º segundo é uma
  frase em inglês do que acontece ("4 SPINNING BARS MULTIPLYING BALLS", "1 BALL vs 108 BLOCKS").
  Contador ao vivo ("N BALLS"). Embaixo: "Follow for more!". Tela final:
  "N BALLS!", "Liked it? FOLLOW US!", "Share it with a friend" (já no motor).
- Nada de luz piscando rápido (epilepsia).
- **Som (estudo das referências, 10/11 — já no motor `make_audio`)**: impacto/whoosh no 1º
  segundo, batida (bumbo+chimbal) que ACELERA de ~96 a ~180 BPM até o fim, estalos de bolinhas
  ficando mais densos, pausa de 0,25 s antes do boom, depois boom forte + brilhos. Não tirar.

## Estilo visual (a partir dos vídeos de 2026-11-10) — referência que o dono mandou
Inspirado em canais de "satisfying physics" (ex.: o que o dono mostrou, 28–30 mi de views),
mas com tudo próprio (nunca copiar nome, logo ou vídeo de ninguém):
- `theme="minimal"`: fundo preto, arena com linha branca fina, sem brilho neon, sem "+N",
  contador discreto, texto curto. Mecanismos cinza-prata.
- Manchas de cor: `palette="manchas"`, `inherit=True` (bolinha nova herda a cor da mãe),
  `seeds=3..4` (começa com bolinhas de cores bem diferentes) → blocos de cor disputando espaço.
- Mecanismos grandes e variados: roda de raios (`minimal_roda_raios`, vários `half` no centro),
  moldura retangular girando, quadrado vazado girando no centro, semicírculos nas laterais,
  bastões que passam da borda, portais multiplicadores (x2/x5/x20) — criar código novo
  quando precisar, testar com preview antes.
- Usar o estilo minimalista na maioria dos 8 vídeos do dia, intercalando com os coloridos.

## Motor (pasta `motor/`)
- `bolas.py` — MOTOR ATUAL. Configs em `CONFIGS` (shape: circle, square, hexagon, octagon,
  triangle, heart, rounded; mech: `rods` (bastões: lista de {off, L, w, phase}), `gears`
  (2 engrenagens), `arcs` (arcos que se fecham); palette: vivo/neon/doce; scale: maior/menor;
  hook; t_full ~30; mech `bounce` = sem mecanismo, quica com gravidade; rods com `half: True`
  = ponteiro de relógio, `marks` = marcas de hora). Uso:
  `python3 bolas.py search <nome> 0 3` → mostra cheio/explode/final por semente;
  `python3 bolas.py preview <nome> <semente> 30 300 600 900 1100` → PNGs para conferir;
  `python3 bolas.py render <nome> <semente>` → `out_<nome>/<nome>.mp4` (~8–15 min cada; a
  máquina tem 2 CPUs: rode no máximo 2 renders ao mesmo tempo, em segundo plano).
  Bastões: pivôs com distância > soma dos comprimentos (L) para não se cruzarem.
- `bolas.py` também tem `mech="pieces"` (classe `Pieces`): polígonos metálicos com espessura que giram
  (`w`, `phase`, `off`) e/ou batem como PRENSA (`press=dict(dir, amp, period, ph)`); `fill=True` = peça
  sólida, `closed=False` = linha aberta. Formatos extras: `star`, `pentagon`, `diamond`. Opções de
  tema: `bg_min` (cor do fundo minimal), `grid=True`, `num_only=True` (contador só número).
  Moldura fechada prende bolinhas dentro e deixa buraco fora: deixe aberturas.
- `duelo_itens.py` — DUELO COM ITENS (bandeiras, barras de vida, itens, virada scriptada):
  `python3 duelo_itens.py sim|preview|render usa_brasil [frames…]`.
- `quebra.py` — QUEBRA-QUADRADOS: 1 bolinha vs quadrados numerados; quadrado quebrado vira
  bolinha; no fim 3 rodadas dobrando e explosão. `python3 quebra.py tune <nome>` (acha o fator
  k para limpar em ~32 s; a busca vai só até k=3 — se limpar cedo, teste k maiores à mão, ex.: quebra_circulo k=4.4), `preview <nome> <k> frames…`, `render <nome> <k>`.
- `arena.py`, `auto.py`, `royale*.py`, `legado/` — estilos antigos (duelo/royale), só referência.
- Fontes: `/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf`; emojis:
  `/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf`.
- Rede: o ambiente só acessa GitHub. Não tente enviar arquivos para outros sites.

## Hospedagem do vídeo (para o Metricool baixar)
- Vídeos vão para o branch **`midia`**, recriado do zero a cada noite (órfão + push forçado),
  para o repositório não crescer:
  `git checkout --orphan midia && git rm -rq --cached . && <copiar vídeos para v/> && git add v && git commit && git push -f origin midia`
- Link público de cada vídeo:
  `https://raw.githubusercontent.com/santoscruzia062-prog/videos-diarios/midia/v/<arquivo>.mp4`
- O Metricool baixa e guarda uma cópia própria no momento do agendamento (o campo `media` da
  resposta vira `static.metricool.com/...`). Depois disso o arquivo no GitHub pode sumir.
- Nunca commite vídeos no `main`.

## Metricool (conector)
- Marca: **mente_digital_x**, `blogId` = **7950970**, fuso **America/Sao_Paulo**.
- Redes: instagram (REEL, showReelOnFeed true), tiktok (exige `tiktokData.title`),
  youtube (`type: short`, `title`, `madeForKids: false`, `privacy: public`, category ENTERTAINMENT).
- `autoPublish: true`, `draft: false`. Data no formato `2026-11-09T07:00:00-03:00` e
  `publicationDate: {dateTime, timezone: "America/Sao_Paulo"}`.
- Antes de agendar, confira com `getScheduledPosts` se o horário já não tem post (evitar duplicar).
  Rascunhos (`draft: true`) não publicam e não ocupam horário — ignore-os.
- Se o Metricool recusar por limite do plano ou outro erro: não insista em loop; registre no
  `historico.md` e no resumo final.

## Legendas — EM INGLÊS (pedido do dono, 08/11: os perfis com engajamento usam inglês)
- Legenda, título do YouTube, `tiktokData.title` e hashtags **em inglês**, relacionados ao vídeo.
  (O texto DENTRO do vídeo é bilíngue: inglês grande + português pequeno embaixo.)
- Instagram/TikTok: 1ª linha = gancho com emoji (what happens); 2ª = como termina (💥, sem dar
  número); 3ª = CTA variado ("Follow for more ❤️", "Send this to someone you love ❤️",
  "Follow to see the next one ❤️"). **No máximo 3 hashtags (NUNCA mais que 3)** em inglês, relacionadas ao vídeo,
  e **sem repetir** hashtags entre os posts do dia (ex.: #satisfying #physics #bouncingball;
  #brickbreaker #satisfyingvideo #blocks; #gears #triangle #funphysics).
- YouTube: título curto em inglês com emojis, sem hashtags (≤ 100 caracteres); tags em inglês.
- TikTok `title`: versão curta do título em inglês.
- Nunca entregue o número final na legenda.
- **Escrever NATURAL, como gente** (pedido do dono, 10/11): frases curtas, minúsculas, tom de
  conversa, poucos emojis (0–2), nada de fórmula repetida tipo "…until it EXPLODES 💥". Ex.:
  "it starts at 10 rpm and just keeps getting faster / by the end it's pure chaos 😅 /
  send this to someone who needs to relax". Variar o CTA (tag someone, sound on, good night…).

## Commits
Termine as mensagens de commit com:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
