-- Techspeed FuelTech Velopark: the start gantry's board and lamps in BeamNG.drive.
--
-- As the Custom Shaders Patch script does in Assetto Corsa: the board shows ten lap times in
-- its rows (places 1 to 4 on the left, 5 and 6 in the middle, 7 to 10 on the right), each as
-- place, number and time, and a clock below the middle rows: the race's time while one runs,
-- the time of day otherwise. A time trial has one car, so the rows are the best laps driven
-- on the layout whose race starts under the gantry, kept between sessions, the number being
-- which lap of its race it was. The five start lamps light one by one through the race's
-- countdown and go out at the start; the flag lamp is green but through the countdown.
--
-- Each character on the board is a shape of its own, swapped as the text changes, as
-- BeamNG's drag strip boards do; the lamps glow by their instance colour, as its traffic
-- signals do. Where everything stands comes from the level's board.json.

local M = {}

local logTag = 'techspeedFueltechVelopark_board'
local LEVEL = 'techspeed_fueltech_velopark'
local CONFIG = '/levels/' .. LEVEL .. '/board.json'
local SAVE = '/settings/techspeed_fueltech_velopark_laps.json'
local GLYPHS = '/levels/' .. LEVEL .. '/art/shapes/board/glyph_'
local SWAPS_PER_FRAME = 6     -- a swap loads a shape: a few a frame, not a whole board at once
local LAMP_STEP = .6          -- seconds between start lamps lighting
local COUNTDOWN_LIMIT = 8     -- seconds a countdown may run before the lamps give up on it
local NEAR = 60               -- metres from a gantry that a countdown is its race's
local LAMP_ON, LAMP_OFF = '1 0.05 0.02 1', '0 0 0 1'
local FLAG_GREEN = '0.05 1 0.15 1'
local ROWS = 10

local config, gantries, group, best
local queue, queued = {}, {}
local active               -- the race running: {race, gantry, layout, laps}
local countdown            -- {gantry, time} through a race's countdown
local oldTrigger

local function ours()
  return getCurrentLevelIdentifier and getCurrentLevelIdentifier() == LEVEL
end

-- ---- characters ---------------------------------------------------------------------------

local function shapeOf(ch)
  local name = ch == ':' and 'colon' or ch == '.' and 'dot' or ch
  return GLYPHS .. name .. '.dae'
end

local function want(cell, ch)
  if not ch:match('^[0-9:.]$') then ch = ' ' end
  if cell.char == ch then return end
  cell.char = ch
  if not queued[cell] then
    queued[cell] = true
    table.insert(queue, cell)
  end
end

local function apply(cell)
  local ch = cell.char
  if ch == ' ' then
    if cell.obj then cell.obj.hidden = true end
    return
  end
  local shape = shapeOf(ch)
  if not cell.obj then
    local o = createObject('TSStatic')
    o:setField('shapeName', 0, shape)
    o:setField('collisionType', 0, 'None')
    o:setField('decalType', 0, 'None')
    o.canSave = false
    o:registerObject(Sim.getUniqueName('velopark_glyph'))
    local p, right, up = cell.pos, vec3(cell.right), vec3(cell.up)
    -- the shape faces its -y, x across and z up: y is up crossed with across
    local rot = quatFromDir(up:cross(right), up)
    o:setPosRot(p[1], p[2], p[3], rot.x, rot.y, rot.z, rot.w)
    o:setScale(vec3(cell.size, cell.size, cell.size))
    group:addObject(o)
    cell.obj, cell.shape = o, shape
  elseif cell.shape ~= shape then
    cell.obj:setField('shapeName', 0, shape)
    cell.shape = shape
  end
  cell.obj.hidden = false
end

local function write(cells, text, from)
  from = from or 1
  for i = 0, #cells - from do
    local k = from + i
    if cells[k] then want(cells[k], text:sub(i + 1, i + 1)) end
  end
end

-- ---- text ---------------------------------------------------------------------------------

local function lapText(t)
  if not t or t <= 0 then return '' end
  return string.format('%d:%06.3f', math.floor(t / 60), t % 60)
end

local function pad(s, n)
  s = s:sub(1, n)
  return s .. string.rep(' ', n - #s)
end

local function showRows(g)
  local laps = (g.layout and best[g.layout]) or {}
  local c = config.cells
  for r = 1, ROWS do
    local e = laps[r]
    local text = e and (pad(tostring(r), c.rank) .. pad(tostring(e.lap or ''), c.number) .. pad(lapText(e.t), c.time)) or ''
    for _, face in ipairs(g.faces) do write(face.rows[r], pad(text, c.rank + c.number + c.time)) end
  end
end

local function clockText(g)
  if active and active.gantry == g and active.race.time then
    local s = math.max(math.floor(active.race.time), 0)
    return string.format('%d:%02d', math.floor(s / 60), s % 60)
  end
  -- BeamNG's day starts at noon
  local tod = core_environment and core_environment.getTimeOfDay and core_environment.getTimeOfDay()
  local h = ((tod and tod.time or 0) * 24 + 12) % 24
  return string.format('%02d:%02d', math.floor(h), math.floor(h % 1 * 60))
end

-- ---- lamps --------------------------------------------------------------------------------

local function colour(name, c)
  local o = scenetree.findObject(name)
  if not o then return end
  o:setField('instanceColor', 0, c)
  if o.updateInstanceRenderData then o:updateInstanceRenderData() end
end

local function lamps(g, lit)
  for i, name in ipairs(g.lamps) do colour(name, i <= lit and LAMP_ON or LAMP_OFF) end
end

local function flags(g, on)
  for _, name in ipairs(g.flags) do colour(name, on and FLAG_GREEN or LAMP_OFF) end
end

-- ---- laps ---------------------------------------------------------------------------------

local function save()
  jsonWriteFile(SAVE, best, true)
end

local function addLap(layout, t, lap)
  best[layout] = best[layout] or {}
  local laps = best[layout]
  table.insert(laps, {t = t, lap = lap, date = os.time()})
  table.sort(laps, function(a, b) return a.t < b.t end)
  while #laps > ROWS do table.remove(laps) end
  save()
  for _, g in ipairs(gantries) do
    if g.layout == layout then showRows(g) end
  end
end

local function gantryNear(pos, within)
  local found, dist
  for _, g in ipairs(gantries) do
    local d = pos:distance(vec3(g.centre))
    if d < within and (not dist or d < dist) then found, dist = g, d end
  end
  return found
end

local function playerState(race)
  local id = be:getPlayerVehicleID(0)
  if race.states[id] then return race.states[id] end
  for _, s in pairs(race.states) do
    if not s.isAiVeh then return s end
  end
end

local function layoutOf(g)
  local mm = gameplay_missions_missionManager
  local id = mm and mm.getForegroundMissionId and mm.getForegroundMissionId()
  local lay = type(id) == 'string' and id:match('/%d+%-([^/]+)$')
  return lay or g.default
end

local function takeLaps()
  if not active then return end
  local s = playerState(active.race)
  if not s or not s.historicTimes then return end
  while active.laps < #s.historicTimes do
    active.laps = active.laps + 1
    local lap = s.historicTimes[active.laps]
    if lap.duration and lap.duration > 0 then addLap(active.layout, lap.duration, (lap.lap or active.laps - 1) + 1) end
  end
end

local function onRaceStarted(data)
  if not gantries or not data or not data.race then return end
  local race = data.race
  local ok, start = pcall(function() return race.path.pathnodes.objects[race.path.startNode].pos end)
  local g = ok and start and gantryNear(vec3(start), 40)
  if not g then return end
  active = {race = race, gantry = g, layout = layoutOf(g), laps = 0}
  -- lights out: the race is on
  countdown = nil
  lamps(g, 0)
  flags(g, true)
  if g.layout ~= active.layout then
    g.layout = active.layout
    showRows(g)
  end
end

local function onRacePathnodeReached(data)
  if active and data and data.race == active.race then takeLaps() end
end

local function finish(data)
  if active and data and data.race == active.race then
    takeLaps()
    active = nil
  end
end

-- the countdown: the time trial shows it as flash messages, 3, 2, 1, then the start
local function onFlash(args)
  if not gantries or active then return end
  local first = type(args) == 'table' and args[1]
  local msg = type(first) == 'table' and first[1]
  if not tonumber(msg) or countdown then return end
  local veh = be:getPlayerVehicle(0)
  local g = veh and gantryNear(vec3(veh:getPosition()), NEAR)
  if not g then return end
  countdown = {gantry = g, time = 0}
  flags(g, false)
end

local function hookTrigger()
  if oldTrigger or not guihooks or not guihooks.trigger then return end
  oldTrigger = guihooks.trigger
  guihooks.trigger = function(name, ...)
    if name == 'ScenarioFlashMessage' then
      local ok, err = pcall(onFlash, ...)
      if not ok then log('W', logTag, 'countdown: ' .. tostring(err)) end
    end
    return oldTrigger(name, ...)
  end
end

local function unhookTrigger()
  if oldTrigger then
    guihooks.trigger = oldTrigger
    oldTrigger = nil
  end
end

-- ---- the level ----------------------------------------------------------------------------

local function teardown()
  unhookTrigger()
  if group then
    pcall(function() group:deleteAllObjects() end)
    pcall(function() group:delete() end)
  end
  group, gantries, config, active, countdown = nil, nil, nil, nil, nil
  queue, queued = {}, {}
end

local function setup()
  teardown()
  config = jsonReadFile(CONFIG)
  if not config or not config.gantries then
    log('W', logTag, 'no ' .. CONFIG)
    return
  end
  best = jsonReadFile(SAVE) or {}
  group = createObject('SimGroup')
  group:registerObject('velopark_board')
  group.canSave = false
  gantries = config.gantries
  for _, g in ipairs(gantries) do
    g.layout = g.default
    lamps(g, 0)
    flags(g, true)
    showRows(g)
  end
  hookTrigger()
  log('I', logTag, string.format('%d gantries', #gantries))
end

local function onUpdate(dtReal, dtSim)
  if not gantries then return end
  if countdown then
    countdown.time = countdown.time + (dtReal or 0)
    if countdown.time > COUNTDOWN_LIMIT then
      lamps(countdown.gantry, 0)
      flags(countdown.gantry, true)
      countdown = nil
    else
      lamps(countdown.gantry, math.min(#countdown.gantry.lamps, math.floor(countdown.time / LAMP_STEP) + 1))
    end
  end
  for _, g in ipairs(gantries) do
    local text = clockText(g)
    if text ~= g.clock then
      g.clock = text
      for _, face in ipairs(g.faces) do write(face.clock, text) end
    end
  end
  for _ = 1, SWAPS_PER_FRAME do
    local cell = table.remove(queue, 1)
    if not cell then break end
    queued[cell] = nil
    apply(cell)
  end
end

local function onClientStartMission()
  if ours() then setup() else teardown() end
end

local function onClientEndMission()
  teardown()
end

local function onExtensionLoaded()
  if ours() then setup() end
end

local function onExtensionUnloaded()
  teardown()
end

M.onClientStartMission = onClientStartMission
M.onClientEndMission = onClientEndMission
M.onExtensionLoaded = onExtensionLoaded
M.onExtensionUnloaded = onExtensionUnloaded
M.onUpdate = onUpdate
M.onRaceStarted = onRaceStarted
M.onRacePathnodeReached = onRacePathnodeReached
M.onRaceComplete = finish
M.onRaceAborted = finish

-- for the console: techspeedFueltechVelopark_board.clearLaps('vp_1500'), or every layout with no name
M.clearLaps = function(layout)
  if layout then best[layout] = nil else best = {} end
  save()
  if gantries then for _, g in ipairs(gantries) do showRows(g) end end
end

return M
