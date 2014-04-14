function d = distance(theta, X, y)
  % returns h_theta(x) - y
  d = (X * repmat(theta, 1, length(y))(:,1)) - y;
end
