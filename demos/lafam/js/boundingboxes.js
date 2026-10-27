// Bounding boxes around connected areas of grid cells with the same group.

// grid: rows of group ids; returns the areas of more than one cell, skipping the given ids
function findAreas(grid, skipNumbers = []) {
  const rows = grid.length;
  const cols = grid[0].length;
  const visited = Array.from({ length: rows }, () => Array(cols).fill(false));
  const skip = new Set(skipNumbers);
  const areas = [];

  function dfs(r, c, number, area) {
    area.squares.push([r, c]);
    visited[r][c] = true;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc] && grid[nr][nc] === number) {
        dfs(nr, nc, number, area);
      }
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (visited[r][c] || skip.has(grid[r][c])) continue;
      const area = { groupId: grid[r][c], squares: [] };
      dfs(r, c, grid[r][c], area);
      areas.push(area);
    }
  }
  return areas.filter((area) => area.squares.length > 1);
}

function findBoundingBoxes(areas) {
  return areas.map((area) => {
    const rows = area.squares.map(([r]) => r), cols = area.squares.map(([, c]) => c);
    return {
      groupId: area.groupId,
      topLeft: [Math.min(...rows), Math.min(...cols)],
      bottomRight: [Math.max(...rows), Math.max(...cols)],
    };
  });
}

// draws the boxes on a canvas showing a grid of gridSize x gridSize cells; nameOf(groupId) labels them
function drawBoundingBoxes(boxes, canvas, gridSize, nameOf, color = "#ff0000") {
  const ctx = canvas.getContext("2d");
  const side = Math.min(canvas.width, canvas.height);
  const cell = side / gridSize;
  const border = Math.round(Math.max(5, Math.min(11, cell / 16)));
  const fontSize = Math.max(16, side / 25);

  for (const box of boxes) {
    const [top, left] = box.topLeft;
    const [bottom, right] = box.bottomRight;
    const x = Math.max(Math.round(left * cell), border / 2);
    const y = Math.max(Math.round(top * cell), border / 2);
    const width = Math.min(Math.round((right - left + 1) * cell), side - x - border / 2);
    const height = Math.min(Math.round((bottom - top + 1) * cell), side - y - border / 2);

    ctx.strokeStyle = color;
    ctx.lineWidth = border;
    ctx.strokeRect(x, y, width, height);

    const textX = x + border / 2 + 2, textY = y + border / 2 + 2, maxWidth = width - border - 4;
    ctx.textBaseline = "top";
    ctx.font = `bold ${fontSize}px Arial`;
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 3;
    ctx.fillStyle = color;
    ctx.strokeText(nameOf(box.groupId), textX, textY, maxWidth);
    ctx.fillText(nameOf(box.groupId), textX, textY, maxWidth);
  }
}
