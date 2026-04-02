export type AppRoute = "builder" | "prune";

export function getRouteFromHash(hash: string): AppRoute {
  if (hash === "#/prune") {
    return "prune";
  }

  return "builder";
}

export function routeToHash(route: AppRoute): string {
  return route === "prune" ? "#/prune" : "#/builder";
}

export function navigateToRoute(route: AppRoute) {
  window.location.hash = routeToHash(route);
}

