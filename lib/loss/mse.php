<?php

function mse($Y, $Y_hat) { //Real vs Expected
  assert(count($Y) == count($Y_hat), 'Real vs Expected sample size must be equal');
  foreach($Y as $sample_index => $sample) {
      assert(count($sample) == count($Y_hat[$sample_index]), 'Real vs Expected must have same dimension count');
      $e = 0;
    	foreach($sample as $i => $y)
    		$e += pow(((double)$Y_hat[$sample_index][$i] - (double)$y), 2);
  }
  return $e / (double)(count($Y) * 2.0);
}
