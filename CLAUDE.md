# Rotina diária de vídeos — manual de operação

Este repositório é a "memória" da rotina. Toda execução começa do zero: leia este arquivo
e o `historico.md` inteiros antes de criar qualquer coisa, e atualize o `historico.md` no final.

## O que a rotina faz (toda noite, ~21h, horário de Brasília)
1. Pesquisa rapidamente (WebSearch) o que está viralizando em vídeos curtos do tipo
   simulação / desafio / duelo / ilusão / participação. Use como inspiração, não copie.
2. Escolhe **6 vídeos diferentes** para o DIA SEGUINTE, sem repetir formato+tema dos últimos
   7 dias (ver `historico.md`). Misture: identidade (países, times, signos, meses, comidas…),
   suspense (escapa? transborda?), participação (ache o diferente, quantos tem?).
3. Cria os vídeos, confere quadros-chave (início, meio, cheio, explosão, final) olhando as
   imagens, e só segue se estiverem bons.
4. Envia os vídeos para o branch `midia` (ver "Hospedagem") e agenda no Metricool:
   **7h, 9h, 11h, 13h, 16h, 20h** do dia seguinte, nas 3 redes.
5. Limpa o branch `midia`, atualiza `historico.md` (branch `main`) e termina com um resumo curto.
   Não precisa de aprovação do dono: faça bem feito e agende direto.

## Padrão de qualidade (pedido do dono — obrigatório)
- Vertical 720×1280, 30 fps, ~40–50 s, H.264 + AAC, som próprio sintetizado (nunca música
  com direitos autorais, nunca logos/escudos de marcas — times só com cores e nome).
- **Gancho no 1º segundo**: frase/pergunta que convida a participar ("ESCOLHA SEU TIME",
  "Quem vence?", "Ache o diferente"). Ritmo constante até o fim, nada parado.
- **Finais com explosão**: o espaço enche TOTALMENTE; depois disso mais **3 rodadas** com a
  borda balançando e tremendo cada vez mais (tique-taque acelerando), e só então explode.
- Interação com quem assiste: placar ao vivo, "VIROU!", contagem regressiva, convite a comentar,
  tela final ("VENCEU!/CAMPEÃO!", "Você acertou?").
- Emojis dos dois lados precisam ter cores bem diferentes (ex.: 😍 x 😡 é ruim; 😍 x 👿 é bom).
- Nada de luz piscando rápido (epilepsia).
- Inovar: a cada semana, pelo menos 1 formato realmente novo (código novo), testado com cuidado.

## Motores prontos (pasta `motor/`)
- `arena.py` — DUELO 2 times em qualquer formato (square, rect, circle, octagon, diamond, star,
  heart, pentagon). Configs em `CONFIGS`; adicione novas entradas (formato, braços, nomes,
  emojis, cores BGR, sons, fundo). Uso:
  `cd motor && python3 auto.py <nome>` → testa sementes, escolhe a partida mais disputada e
  renderiza `out_<nome>/duelo_<nome>.mp4`. (Crescimento controlado para lotar ~33 s, fator de
  virada, 3 rodadas finais, explosão, tela do vencedor.)
  Sons disponíveis por time: roar, howl, love, angry, sun, moon, kick, bounce, bark, meow, pop,
  chomp, ice, coin.
- `royale.py` — BATALHA ROYALE de 8 países (eliminação a cada 5 s, ranking ao vivo, campeão).
  `python3 royale.py search 0 8` → escolha a semente; `python3 royale.py render <semente>`.
- `royale_times.py` — mesma batalha com 8 times brasileiros (distintivos só de cores).
  Para outros temas, copie e troque `TEAMS`, `TITLE`, textos e `GENDER`.
- `legado/` — versões antigas (arcos giratórios, corações, dinheiro 🤑 no quadrado, hexágono)
  com ideias reaproveitáveis.
- Fontes: `/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf`; emojis coloridos:
  `/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf` (inclui bandeiras).
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
- Instagram/TikTok: 1ª linha = gancho com emoji; 2ª = o que acontece; 3ª = chamada para
  comentar ("Comenta seu palpite ANTES de ver o final 👇"); depois 8–10 hashtags:
  tema (#leaovslobo, #qualpaissobrevive), formato (#duelo, #batalharoyale, #simulacao,
  #desafio) e alcance (#viral, #fyp, #foryou).
- YouTube: título curto com emojis + `#shorts` (≤ 100 caracteres); tags do tema.
- TikTok `title`: versão curta do título.
- Nunca entregue o resultado na legenda.

## Commits
Termine as mensagens de commit com:
```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```
