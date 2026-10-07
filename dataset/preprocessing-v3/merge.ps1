# Uses Excel to copy every sheet of $newFile to the end of $target, recalculates, saves, and prints a check.
param([string]$target, [string]$newFile)
$x = New-Object -ComObject Excel.Application
$x.DisplayAlerts = $false
$x.Visible = $false
try {
  $dst = $x.Workbooks.Open($target)
  $src = $x.Workbooks.Open($newFile)
  $last = $dst.Worksheets.Item($dst.Worksheets.Count)
  $src.Worksheets.Copy([Type]::Missing, $last)   # copy all new sheets together so cross-sheet formulas stay internal
  $src.Close($false)
  $links = $dst.LinkSources(1)
  if ($links) { foreach ($l in $links) { $dst.ChangeLink($l, $dst.FullName, 1) } }
  $x.CalculateFull()
  $dst.Save()
  foreach ($ws in $dst.Worksheets) { "{0} | rows {1} | visible {2}" -f $ws.Name, $ws.UsedRange.Rows.Count, $ws.Visible }
  "external links left: " + [bool]$dst.LinkSources(1)
  $st = $dst.Worksheets.Item("Stats v3")
  for ($r = 1; $r -le 48; $r++) {
    $line = @(); for ($c = 1; $c -le 8; $c++) { $line += $st.Cells.Item($r, $c).Text }
    if (($line -join '').Trim()) { ($line -join ' | ') }
  }
  "Old Stats G6 = " + $dst.Worksheets.Item("Stats").Range("G6").Text + " ; G15 = " + $dst.Worksheets.Item("Stats").Range("G15").Text
  "Tables on sheet 1: " + $dst.Worksheets.Item(1).ListObjects.Count
  $dst.Close($true)
} catch { "ERROR: " + $_.Exception.Message } finally { $x.Quit() }
