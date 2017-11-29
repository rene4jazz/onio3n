<?php

function initConv( $input_size,
                     $filter_size,
                     $filter_count,
                     $stride = 1
                     //$zero_padding = 0
                   ) {
  $layer = [
    'type' => 'Conv',
    'input_size' => $input_size_i,
    'filter_size' => $filter_size,
    'filter_count' => $filter_count,
    'stride' => $stride,
    //'padding' => $padding,
    'filters' => new SplFixedArray($filter_size * $filter_count),
    'biases' => new SplFixedArray($filter_count),
    'output_size' =>
    // 'output' => //new SplFixedArray( //  (W − F + 2P) / S + 1 http://cs231n.github.io/convolutional-networks/
        (
          ($input_size_i) /* W */
          - ($filter_size) /* F */
          // - (2 * $padding )) /* P */
          /
          $stride /* S */ + 1
        )
        * $filter_count, // times filter count :)
     //)
  ];
}

function forwardConv($X, &$layer, $training_mode = false) {

  $X = (get_class($X) == 'SplFixedArray' ) ? $X : SplFixedArray::fromArray($X);

  assert($X->getSize() == $layer['input_size'], 'Input size mistmatch on conv layer, check configuration' );

  $Y = new SplFixedArray($layer['output_size']);
  $o_size = $layer['output']->getSize();
  $s = $f_count = $layer['stride'];
  $f_size = $layer['filter_size'];
  $f_count = $layer['filter_count'];
  $f_size = $f_offset * $f_count;
  $input_size = $X->getSize();

  for ($curr_f = 0; $curr_f < $f_count; $curr_f++) {
    for ($input_patch_index = 0; $input_patch_index < $input_size; $input_patch_index += $stride) {
      $y = 0;
      for ($f_index = 0; $f_index < $f_size; $f_index++) {
        $y += $X[$f_index + ($o_index / $f_offset)] * $layer['filters'][$f_size*$curr_f + $f_index];
      }
      $y += $layer['biases'][$curr_f];
      $Y[]
    }
  }

  for ($o_index = 0; $o_index < $o_size; $o_index++) {
    for ($curr_f = 0; $curr_f < $f_count; $curr_f++) {
      $y = 0;
      $current_input_offset = 0;
      for ($f_index = 0; $f_index < $f_offset; $f_index++) {
        $y += $X[$f_index + ($o_index / $f_offset)] * $layer['filters'][$f_index/$f_offset];

      }
    }
  }

  $
  $W = $layer['filters'];

  if ($training_mode) {
    $layer['grad'] = ['input' => $X, 'delta' => []];
  }

  foreach ($W as $unit_i => $unit) {

    $y = 0;
    foreach($unit as $w_i => $w) {
      $y += ((double) $X[$w_i]) * ((double)$w);
    }

    $Y[] = $y + $B[$unit_i];
  }
  return $Y;
}

function backwardConv($delta, &$layer, $lr) {
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
