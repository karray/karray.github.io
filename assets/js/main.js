let hexValues = ["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e"];

function getRandomColor() {
    let a = ''
    for ( let i = 0; i < 6; i++ ) {
        let x = Math.round( Math.random() * 14 );
        let y = hexValues[x];
      a += y;
    }
    return '#'+a;
}

function generateGradient() {
    return `linear-gradient(${Math.round( Math.random() * 360 )}deg, ${getRandomColor()}, ${getRandomColor()})`;
}