--[[
book.lua — pandoc Lua filter for the PicoGPT book (web AND print share its output).

Conventions (BRIEF §7). Input: one chapter, `book/chapters/NN-slug.md`.
  # Title {#ch-id}           chapter title → metadata (the page renders the opening)
  ::: claim                  the chapter's one-sentence claim
  ::: names                  the three-names table (maths | physics | ML)
  ::: aside                  margin note; set beside the block that PRECEDES it
  ^[...] / [^n]              numbered sidenote
  ::: {.figure #fig-id k=v}  figure slot; the div's content is the caption
  ::: {.code file=… fn=…}    code excerpt pulled from the Julia source at build time
  ::: exercise               numbered exercise
  @fig-id @sec-id @ch-id     cross references (numbers come from the book index)
  $…$, $$…$$                 maths (LaTeX), rendered at build time to MathML

Metadata in: -M chapter=<n>  -M index=<path to JSON {figures:{id:num}, sections:{id:num}, chapters:{id:{num,slug}}}>
Output: an HTML fragment. Machine-readable markers (HTML comments / custom elements)
are consumed by src/lib/server/book.js:
  <!--BOOKMETA {...}-->            chapter metadata (title, id, claim)
  <!--FIG {...}--> caption <!--/FIG-->  figure slots (split into segments)
  <!--CODE {...}-->                code excerpts
  <tex-m>…</tex-m>, <tex-d>…</tex-d>  maths source (HTML-escaped)
]]

local chapter = '0'
local index = { figures = {}, sections = {}, chapters = {} }
local chapter_id, chapter_title, claim_html = nil, nil, nil
local fig_count, ex_count, sec_count, note_count = 0, 0, 0, 0

local function esc(s)
  return (s:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;'):gsub('"', '&quot;'))
end

local function json(v) return pandoc.json.encode(v) end

local function html(blocks)
  return pandoc.write(pandoc.Pandoc(blocks), 'html5', { wrap_text = 'none' })
end

local function inline_html(inlines)
  return pandoc.write(pandoc.Pandoc({ pandoc.Plain(inlines) }), 'html5', { wrap_text = 'none' })
end

-- ── metadata ──────────────────────────────────────────────────────────────
function Meta(meta)
  if meta.chapter then chapter = pandoc.utils.stringify(meta.chapter) end
  if meta.index then
    local path = pandoc.utils.stringify(meta.index)
    local fh = io.open(path, 'r')
    if fh then
      local ok, data = pcall(pandoc.json.decode, fh:read('a'), false)
      fh:close()
      if ok and type(data) == 'table' then index = data end
    end
  end
  return meta
end

-- ── inlines ───────────────────────────────────────────────────────────────
local function math_inline(el)
  local tag = el.mathtype == 'DisplayMath' and 'tex-d' or 'tex-m'
  return pandoc.RawInline('html', '<' .. tag .. '>' .. esc(el.text) .. '</' .. tag .. '>')
end

local function ref_text(id)
  if id:match('^fig%-') then
    local n = index.figures and index.figures[id]
    return n and ('Figure ' .. n) or nil, n
  elseif id:match('^sec%-') then
    local n = index.sections and index.sections[id]
    return n and ('§' .. n) or nil, n
  elseif id:match('^ch%-') then
    local c = index.chapters and index.chapters[id]
    if c and c.num == '' then return c.title or 'Epilogue', '' end
    return c and ('Chapter ' .. c.num) or nil, c and c.num
  elseif id:match('^ex%-') then
    local n = index.exercises and index.exercises[id]
    return n and ('Exercise ' .. n) or nil, n
  end
  return nil
end

local function ref_href(id)
  local where = index.where and index.where[id]
  if where and where ~= chapter then
    local c = index.bynum and index.bynum[where]
    if c then return '../' .. c .. '/#' .. id end
  end
  if id:match('^ch%-') then
    local c = index.chapters and index.chapters[id]
    if c then return '../' .. c.slug .. '/' end
  end
  return '#' .. id
end

local function cite(el)
  if #el.citations ~= 1 then return nil end
  local c = el.citations[1]
  local id = c.id
  local text, num = ref_text(id)
  if not text then
    if id:match('^(fig|sec|ch|ex)%-') then
      io.stderr:write('[book.lua] chapter ' .. chapter .. ': unresolved reference @' .. id .. '\n')
      return pandoc.RawInline('html', '<span class="xref xref--missing">??' .. esc(id) .. '</span>')
    end
    return nil
  end
  if c.mode == 'SuppressAuthor' then text = tostring(num) end
  local ch = id:match('^ch%-') and index.chapters and index.chapters[id]
  if ch and ch.planned then
    return pandoc.RawInline('html', '<span class="xref xref--planned" title="in preparation">' .. esc(text) .. '</span>')
  end
  return pandoc.RawInline('html', '<a class="xref" href="' .. ref_href(id) .. '">' .. esc(text) .. '</a>')
end

local function note(el)
  note_count = note_count + 1
  local n = tostring(note_count)
  local body = pandoc.utils.blocks_to_inlines(el.content, { pandoc.Space() })
  local inner = inline_html(body):gsub('^<p>', ''):gsub('</p>%s*$', '')
  return pandoc.RawInline('html',
    '<sup class="sn-ref" id="snref-' .. chapter .. '-' .. n .. '">' .. n .. '</sup>' ..
    '<span class="sidenote" role="note"><span class="sn-num">' .. n .. '</span> ' .. inner .. '</span>')
end

-- ── blocks ────────────────────────────────────────────────────────────────
local function has(el, cls) return el.classes and el.classes:includes(cls) end

local function attrs_table(el)
  local t = {}
  for k, v in pairs(el.attributes) do t[k] = v end
  return t
end

local function header(el)
  if el.level == 1 then
    chapter_title = pandoc.utils.stringify(el.content)
    chapter_id = el.identifier ~= '' and el.identifier or nil
    -- keep the formatted title too (it may contain maths or emphasis)
    index._title_html = inline_html(el.content)
    return {}
  end
  if el.level == 2 and (el.classes:includes('unnumbered') or el.classes:includes('-')) then
    return el
  end
  if el.level == 2 then
    sec_count = sec_count + 1
    local num = chapter .. '.' .. sec_count
    el.attributes['data-num'] = num
    table.insert(el.content, 1, pandoc.RawInline('html', '<span class="secnum">' .. num .. '</span> '))
    return el
  end
  return el
end

local function figure(el)
  fig_count = fig_count + 1
  local id = el.identifier
  local num = (index.figures and index.figures[id]) or (chapter .. '.' .. fig_count)
  local spec = { id = id, num = num, attrs = attrs_table(el) }
  local classes = {}
  for _, c in ipairs(el.classes) do if c ~= 'figure' then table.insert(classes, c) end end
  spec.classes = classes
  local out = { pandoc.RawBlock('html', '<!--FIG ' .. json(spec) .. '-->') }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock('html', '<!--/FIG-->'))
  return out
end

local function code(el)
  local spec = attrs_table(el)
  spec.caption = #el.content > 0 and html(el.content) or nil
  return pandoc.RawBlock('html', '<!--CODE ' .. json(spec) .. '-->')
end

local function exercise(el)
  ex_count = ex_count + 1
  local num = chapter .. '.' .. ex_count
  local id = el.identifier ~= '' and el.identifier or ('ex-' .. chapter .. '-' .. ex_count)
  local title = el.attributes.title
  local head = '<section class="exercise" id="' .. id .. '"><h4 class="exercise__label">' ..
    '<span class="exercise__word">Exercise</span> <span class="exercise__num">' .. num .. '</span>' ..
    (title and (' <span class="exercise__title">' .. esc(title) .. '</span>') or '') .. '</h4>'
  local out = { pandoc.RawBlock('html', head) }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock('html', '</section>'))
  return out
end

local function claim(el)
  claim_html = html(el.content):gsub('^%s*<p>', ''):gsub('</p>%s*$', '')
  return {}  -- the chapter opening renders the claim
end

local function names(el)
  local out = { pandoc.RawBlock('html', '<div class="names" role="group" aria-label="Three names">') }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock('html', '</div>'))
  return out
end

-- Asides are attached to the block that precedes them (top level only):
--   paragraph etc.  →  <div class="noted"> <aside class="note">…</aside> BLOCK </div>
--                      (the aside floats into the margin, aligned with the block's top)
--   a figure        →  the aside moves inside the figure (set in the figure's margin)
local function attach_pass(d)
  local out = {}
  for _, b in ipairs(d.blocks) do
    if b.t == 'Div' and has(b, 'aside') then
      local prev = out[#out]
      if prev and prev.t == 'Div' and has(prev, 'figure') then
        prev.content:insert(b)
      elseif prev and prev.t == 'Div' and has(prev, 'noted') then
        prev.content:insert(#prev.content, b)       -- several notes on one block
      elseif prev and prev.t ~= 'RawBlock' then
        out[#out] = pandoc.Div({ b, prev }, pandoc.Attr('', { 'noted' }))
      else
        table.insert(out, pandoc.Div({ b }, pandoc.Attr('', { 'noted' })))
      end
    else
      table.insert(out, b)
    end
  end
  d.blocks = out
  return d
end

local function aside(el)
  local cls = { 'note' }
  for _, c in ipairs(el.classes) do if c ~= 'aside' then table.insert(cls, c) end end
  local out = { pandoc.RawBlock('html', '<aside class="' .. table.concat(cls, ' ') .. '">') }
  for _, b in ipairs(el.content) do table.insert(out, b) end
  table.insert(out, pandoc.RawBlock('html', '</aside>'))
  return out
end

local function div(el)
  if has(el, 'figure') then return figure(el) end
  if has(el, 'code') then return code(el) end
  if has(el, 'exercise') then return exercise(el) end
  if has(el, 'claim') then return claim(el) end
  if has(el, 'names') then return names(el) end
  if has(el, 'aside') then return aside(el) end
  return nil
end

-- ── document ──────────────────────────────────────────────────────────────
local function doc(d)
  local meta = {
    id = chapter_id, title = chapter_title, titleHtml = index._title_html,
    claim = claim_html, chapter = chapter,
    counts = { figures = fig_count, exercises = ex_count, sections = sec_count, notes = note_count },
  }
  table.insert(d.blocks, 1, pandoc.RawBlock('html', '<!--BOOKMETA ' .. json(meta) .. '-->'))
  return d
end

return {
  { Meta = Meta },
  { Math = math_inline, Cite = cite, Note = note },
  { Pandoc = attach_pass },
  { Header = header, Div = div },
  { Pandoc = doc },
}
