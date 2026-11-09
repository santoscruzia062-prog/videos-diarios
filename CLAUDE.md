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
- **NÃO pedir para adivinhar** quantas bolinhas (o dono não quer). Gancho no 1º segundo é uma
  frase do que acontece ("4 BASTÕES MULTIPLICANDO BOLINHAS", "1 BOLINHA vs 108 QUADRADOS").
  Contador ao vivo de bolinhas. Embaixo: "Segue a página pra ver mais!". Tela final:
  "N BOLINHAS!", "Gostou? SEGUE A PÁGINA!", "Curte e compartilha com alguém".
- Nada de luz piscando rápido (epilepsia).

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
- `quebra.py` — QUEBRA-QUADRADOS: 1 bolinha vs quadrados numerados; quadrado quebrado vira
  bolinha; no fim 3 rodadas dobrando e explosão. `python3 quebra.py tune <nome>` (acha o fator
  k para limpar em ~32 s), `preview <nome> <k> frames…`, `render <nome> <k>`.
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

## Legendas (português do Brasil)
- Instagram/TikTok: 1ª linha = gancho com emoji (o que acontece); 2ª = como termina (💥 sem dar
  número); 3ª = "Segue a página pra ver mais ❤️" (variar a frase). **No máximo 3 hashtags**,
  relacionadas ao vídeo, e **não repetir** as mesmas hashtags entre os posts do dia.
- YouTube: título curto com emojis, sem hashtags no título (≤ 100 caracteres); tags do tema.
- TikTok `title`: versão curta do título.
- Nunca entregue o número final na legenda.

## Commits
Termine as mensagens de commit com:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
