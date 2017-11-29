<?php

function forwardSigmoid($X, &$layer, $training_mode = false) {
  $Y = [];
  foreach($X as $x){
    $Y[] = 1 / (1 + pow(M_E, -$x));
  }
  if ($training_mode) {
    $layer['grad'] = ['output' => $Y, 'delta' => []];
  }
  return $Y;
}

function backwardSigmoid($delta, &$layer, $lr) {
  $Y = [];
  $outputs = $layer['grad']['output'];
  foreach ($delta as $i => $x) { // computes derivate and applies chain rule
    $Y[$i] = $x * $outputs[$i] * (1 - $outputs[$i]);
  }
  $layer['grad']['delta'] = $Y;
  return $Y;
}
