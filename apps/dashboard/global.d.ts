declare module 'react' {
  export type ReactNode = any;
}

declare const process: {
  env: {
    DASHBOARD_API_BASE_URL?: string;
    [key: string]: string | undefined;
  };
};

declare namespace JSX {
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}
