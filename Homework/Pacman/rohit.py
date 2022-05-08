  currentPacmanPosition = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  foodAsList = food.asList()
  ghostStates = currentGameState.getGhostStates()
  huntingGhosts = []
  scaredGhosts = []
  scaredTimes=[]
  
  for ghost in ghostStates:
    if ghost.scaredTimer:
      scaredTimes.append(ghost.scaredTimer)
      scaredGhosts.append(ghost)
    else:
      huntingGhosts.append(ghost)
      
  capsules=currentGameState.getCapsules()
  remainingFood=len(foodAsList)
  remainingCapsules=len(capsules)
  currentScore = currentGameState.getScore()
  distToClosestFood = float("inf")
  invDistanceToClosestFood=0
  for item in foodAsList:
    dist = util.manhattanDistance(currentPacmanPosition, item)
    if dist < distToClosestFood:
      distToClosestFood = dist

  if distToClosestFood>0:
    invDistanceToClosestFood=1/distToClosestFood
  if len(foodAsList)<3:
    invDistanceToClosestFood=100000
  if len(foodAsList)==1:
    invDistToClosestFood=500000
  distToClosestCapsules = float("inf")
  invDistToClosestCapsule=0
  if remainingCapsules == 0:
    distToClosestCapsules = 0
  for item in capsules:
    dist = util.manhattanDistance(currentPacmanPosition, item)
    if dist < distToClosestCapsules:
      distToClosestCapsules = dist
  if distToClosestCapsules>0:
    invDistToClosestCapsule=1/distToClosestCapsules
  distToHuntingGhost=float("inf")
  for ghost in huntingGhosts:
    dist = util.manhattanDistance(currentPacmanPosition, ghost.getPosition())
    if dist < distToHuntingGhost:
      distToHuntingGhost = dist
  #distToHuntingGhost=max(3,distToHuntingGhost)
  if len(scaredGhosts) == 0:
    distToScaredGhost = 0
    scaredTime=0
  else:
    distToScaredGhost = float("inf")
    for ghost in scaredGhosts:
      dist = util.manhattanDistance(currentPacmanPosition, ghost.getPosition())
      if dist < distToScaredGhost:
        distToScaredGhost = dist
    scaredTime=scaredTimes[0]
  invDistToHuntingGhost = 0
  if distToHuntingGhost > 0:
    invDistToHuntingGhost = 1.0 / distToHuntingGhost
  invDistToScaredGhost = 0
  if distToScaredGhost > 0:
    invDistToScaredGhost = 1.0 / distToScaredGhost
  score=currentGameState.getScore()\
        -2*invDistToHuntingGhost\
        +15*scaredTime*invDistToScaredGhost\
        -2*remainingFood\
        -3*invDistToClosestCapsule\
        -1*distToClosestFood
  return score
  #return currentGameState.getScore()
  # END_YOUR_CODE
