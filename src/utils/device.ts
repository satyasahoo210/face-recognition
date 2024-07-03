export const isiOS = () => {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }
  
  export const isAndroid = () => {
    return /Android/i.test(navigator.userAgent);
  }
  
  export const isMobile = () => {
    return isAndroid() || isiOS();
  }