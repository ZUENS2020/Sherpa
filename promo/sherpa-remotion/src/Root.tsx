import React from 'react';
import {Composition} from 'remotion';
import {SherpaPromo, SherpaPromoZh} from './SherpaPromo';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="SherpaPromo"
        component={SherpaPromo}
        durationInFrames={36 * 30}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="SherpaPromoZh"
        component={SherpaPromoZh}
        durationInFrames={36 * 30}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
