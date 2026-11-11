-- Techspeed FuelTech Velopark: the loose cones in BeamNG.drive.
--
-- The cones Assetto Corsa knocks about are BeamNG's own cones, each a vehicle of its cones prop,
-- which a level cannot hold as it holds its shapes. In free roam they are spawned from the
-- level's cones.prefab.json (every layout's cones and distance boards) once it has loaded, as BeamNG's time
-- trials spawn theirs. A race puts away every vehicle but the player's while it runs, these
-- with them, and spawns its own layout's cones from its mission's mainPrefab instead. A race
-- started from the menu loads the level, and BeamNG tells the level it has started once more
-- after the race has begun: these are not spawned then, or they stand in the race's own cones
-- and the two sets burst apart.

local M = {}

local LEVEL = 'techspeed_fueltech_velopark'
local PREFAB = '/levels/' .. LEVEL .. '/cones.prefab.json'
local NAME = 'techspeedFueltechVelopark_cones'

local function ours()
  return getCurrentLevelIdentifier and getCurrentLevelIdentifier() == LEVEL
end

local function remove()
  local prefab = scenetree.findObject(NAME)
  if prefab then
    prefab:unload()
    prefab:delete()
  end
end

local function racing()
  return gameplay_missions_missionManager and gameplay_missions_missionManager.getForegroundMissionId
    and gameplay_missions_missionManager.getForegroundMissionId() ~= nil
end

local function spawn()
  if racing() then return end
  remove()
  if not ours() or not FS:fileExists(PREFAB) then return end
  -- as the time trials' flowgraph spawns its prefab: where it was laid out, not turned
  local prefab = spawnPrefab(NAME, PREFAB, '0 0 0', '0 0 1 0', '1 1 1', false)
  if prefab then
    prefab.canSave = false
  else
    log('E', 'techspeedFueltechVelopark_cones', 'could not spawn ' .. PREFAB)
  end
end

M.onClientStartMission = spawn
M.onClientEndMission = remove
M.onExtensionLoaded = function()
  if ours() then spawn() end
end
M.onExtensionUnloaded = remove

return M
