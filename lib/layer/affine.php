<?php

function initAffine($input_size, $output_size) {
  $rows = [];
  for ($j = 0; $j < $output_size; $j++) {
    $col = [];
    for ($i = 0; $i < $input_size; $i++) {
      $col[] = stats_rand_gen_normal(0, 1);
    }
    $rows[] = $col;
  }

  $col = []; // biases
  for ($i = 0; $i < $output_size; $i++) {
    $col[] = stats_rand_gen_normal(0.5, 0.5);
  }
  $rows[] = $col;
  return $rows;
}

function forwardAffine($X, &$layer, $training_mode = false) {
  assert(key_exists('weights', $layer), 'Affine layer expects weights being defined in layer');
  $Y = [];
  $W = $layer['weights'];
  $B = array_pop($W); //extracting biases
  assert(count($B) == count($W), 'Number of units and bias vector should be the same size');

  if ($training_mode) {
    $layer['grad'] = ['input' => $X, 'delta' => []];
  }

  foreach ($W as $unit_i => $unit) {
    assert(count($X) == count($unit), 'Input and weight vector should be same size');
    $y = 0;
    foreach($unit as $w_i => $w) {
      $y += ((double) $X[$w_i]) * ((double)$w);
    }

    $Y[] = $y + $B[$unit_i];
  }
  return $Y;
}

function backwardAffine($delta, &$layer, $lr) {
  assert(key_exists('grad', $layer), 'Invalid layer format, missinng grad section');
  assert(key_exists('input', $layer['grad']), 'Invalid layer format, missing input values');
  $inputs = $layer['grad']['input'];

  assert(key_exists('weights', $layer), 'Affine layer expects weights being defined in layer');
  $W = $layer['weights'];
  $W_r = &$layer['weights'];
  $B = array_pop($W); //extracting biases
  assert(count($B) == count($W), 'Number of units and bias vector should be the same size');

  $rows = [];
  $Y = array_fill(0, count($inputs), 0); // ouput delta will be same size as layer's input

  $num_rows = count($W);
  $num_cols = count($inputs);
  for ($i = 0; $i < $num_rows; $i++) { // for every output
    $cols = [];
    for ($j = 0; $j < $num_cols; $j++) { // for every input
      $grad = $inputs[$j] * $delta[$i];
      $cols[$j] = $grad;
      $W_r[$i][$j] -= ((double)$lr * (double)$grad); //weight update
      $Y[$j] += $grad; // accumulate error from every next layer input
    }
    $W_r[count($W)][$i] -= (double)$lr * $delta[$i];
    $rows[] = $cols;
  }

  assert(key_exists('delta', $layer['grad']), 'Invalid layer format, missing delta values');
  $layer['grad']['delta'] = $rows;
  return $Y;
}
